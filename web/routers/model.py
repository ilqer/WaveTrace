import json
import os

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# joblib.load unpickles, so both reads and writes are confined to output/.
ALLOWED_ROOT = os.path.realpath("output")


def _resolve_output_path(path: str) -> str:
    resolved_path = os.path.realpath(path)
    if os.path.commonpath([resolved_path, ALLOWED_ROOT]) != ALLOWED_ROOT:
        raise ValueError(f"path escapes output/: {path}")
    return resolved_path


@router.get("/api/model/weights")
async def model_weights(model: str, mode: str = "weapon"):
    """Per-antenna learned CNN channel weights (L2 norms of first conv filters, normalized)."""
    from wavetrace.adapters.recognition.session import open_session
    from wavetrace.recognition.explain import cnn_channel_weights
    try:
        safe_model_path = _resolve_output_path(model)
    except ValueError:
        return {"error": "model path must be inside output/ and must not escape it"}
    try:
        session = open_session(mode, safe_model_path)
        per_antenna_weights = cnn_channel_weights(session.head)
        return {"per_antenna": per_antenna_weights.tolist() if per_antenna_weights is not None else None}
    except Exception as error:
        return {"error": str(error)}


@router.get("/api/fusion/weights")
async def fusion_weights(path: str):
    """Softmax over the saved combiner's coefficients gives per-band trust."""
    import joblib, numpy as np
    try:
        blob = joblib.load(_resolve_output_path(path))
        coefficients = blob["combiner"].coef_.ravel()
        exponentials = np.exp(coefficients - coefficients.max())
        band_weights = exponentials / exponentials.sum()
        return {"bands": blob["band_order"], "weights": [round(float(weight), 3) for weight in band_weights]}
    except Exception as error:
        return {"error": str(error)}


@router.get("/api/weapon/litmus")
async def weapon_litmus(root: str = "data", node: int | None = None, per_link: bool = False):
    """Static σ²[p] separation between clear and weapon captures, per node or per tx→rx link.
    Each row's `hist` holds histogram bins for the client's PDF overlay."""
    from wavetrace.diagnostics import gather_sigma2, json_histogram, key_label, separation, verdict
    try:
        sigma2_by_key = gather_sigma2(root, node, per_link=per_link)
        if not sigma2_by_key:
            return {"error": f"no weapon recordings under {root}/weapon_rec/*/<clear|weapon>/node*/"}

        def _compute_auc(key):
            separation_result = separation(sigma2_by_key[key].get("clear", _empty_array()),
                                           sigma2_by_key[key].get("weapon", _empty_array()))
            return separation_result["auc"] if separation_result else 0.0

        rows = []
        for key in sorted(sigma2_by_key,
                          key=lambda group_key: (-_compute_auc(group_key), key_label(group_key))):
            clear_sigma2 = sigma2_by_key[key].get("clear", _empty_array())
            weapon_sigma2 = sigma2_by_key[key].get("weapon", _empty_array())
            separation_result = separation(clear_sigma2, weapon_sigma2)
            label = key_label(key)
            if separation_result is None:
                rows.append({"label": label, "ok": False, "reason": "need both clear and weapon captures"})
                continue
            rows.append({"label": label, "auc": round(separation_result["auc"], 3),
                         "lower_when_armed": separation_result["lower_when_armed"],
                         "cohens_d": round(separation_result["cohens_d"], 2),
                         "n_clear": separation_result["n_clear"], "n_weapon": separation_result["n_weapon"],
                         "verdict": verdict(separation_result["auc"]),
                         "hist": json_histogram(clear_sigma2, weapon_sigma2)
                         if clear_sigma2.size >= 10 and weapon_sigma2.size >= 10 else None})
        return {"rows": rows, "per_link": per_link}
    except Exception as error:
        return {"error": str(error)}


@router.get("/api/calib/info")
async def calib_info(path: str = "output/calib"):
    """K is max(image_subcarriers)+1: the highest subcarrier index the radio reported during
    calibration. It pins the width the rest of the pipeline must use."""
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        return {"error": f"no calibration at {path} (run Calib first)"}
    try:
        with open(meta_path) as meta_file:
            meta = json.load(meta_file)
        image_subcarriers = meta.get("image_subcarriers") or meta.get("subcarriers") or []
        subcarrier_count = int(max(image_subcarriers)) + 1 if image_subcarriers else 0
        if subcarrier_count <= 96:
            bandwidth_label = "HT20 · 2.4 GHz"
        elif subcarrier_count <= 200:
            bandwidth_label = "HT40 · 2.4 GHz"
        else:
            bandwidth_label = "HT80 · 5 GHz"
        return {
            "K": subcarrier_count,
            "bw_label": bandwidth_label,
            "n_selected": len(meta.get("subcarriers") or []),
            "n_image": len(image_subcarriers),
            "path": path,
        }
    except Exception as error:
        return {"error": str(error)}


def _empty_array():
    import numpy as np
    return np.array([])


class ModelUploadRequest(BaseModel):
    file_b64: str
    dest: str = "output/model.pkl/model.joblib"


@router.post("/api/model/upload")
async def model_upload(request: ModelUploadRequest):
    import base64
    try:
        destination_path = _resolve_output_path(request.dest)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        with open(destination_path, "wb") as model_file:
            model_file.write(base64.b64decode(request.file_b64))
        return {"status": "uploaded", "dest": destination_path}
    except Exception as error:
        return {"error": str(error)}
