import asyncio
import os

from fastapi import APIRouter

from wavetrace.adapters.store.dataset import FEATURES_FILE_NAME
from wavetrace.adapters.store.model_runs import find_model_dir

router = APIRouter()


def _as_model_dir_path(model_path: str) -> str:
    """`<model_dir>/<run_id or current>/model.joblib` as `<model_dir>/model.joblib`, the path every loader
    resolves to the current run; a flat path unchanged."""
    served_dir, file_name = os.path.split(model_path)
    return os.path.join(find_model_dir(served_dir), file_name)


@router.get("/api/paths/scan")
async def scan_paths():
    import glob as _glob

    def _scan():
        calibration_dirs = sorted(set(
            os.path.dirname(matched_path)
            for matched_path in _glob.glob("data/**/meta.json", recursive=True)
                       + _glob.glob("output/**/meta.json", recursive=True)
        ))
        model_files = sorted({
            _as_model_dir_path(matched_path)
            for matched_path in _glob.glob("data/**/model.joblib", recursive=True)
                       + _glob.glob("output/**/model.joblib", recursive=True)
        })
        mesh_roots = sorted({
            os.path.dirname(model_file) for model_file in model_files
            if os.path.basename(os.path.dirname(model_file)).startswith("node")
        })
        feature_paths = (_glob.glob(f"data/**/{FEATURES_FILE_NAME}", recursive=True)
                         + _glob.glob(f"output/**/{FEATURES_FILE_NAME}", recursive=True))
        dataset_dirs = sorted({os.path.dirname(matched_path) for matched_path in feature_paths})
        pooled_dataset_dirs = sorted({os.path.dirname(os.path.dirname(matched_path))
                                      for matched_path in feature_paths
                                      if os.path.basename(os.path.dirname(matched_path)) not in (".", "")})
        return {
            "calibrations": calibration_dirs,
            "models": model_files + [mesh_root for mesh_root in mesh_roots if mesh_root not in model_files],
            "datasets": dataset_dirs + [pooled_dir for pooled_dir in pooled_dataset_dirs
                                        if pooled_dir not in dataset_dirs],
        }

    return await asyncio.to_thread(_scan)


@router.get("/api/paths/browse")
async def browse_path(type: str = "dir", prompt: str = "Select path", ext: str = ""):
    """macOS only: the dialog comes from osascript.
    Returns {"path": "/abs/path"}, or {"path": null} when the user cancels."""
    import subprocess as _subprocess

    def _open_dialog():
        if type == "dir":
            script = f'POSIX path of (choose folder with prompt "{prompt}")'
        else:
            if ext:
                allowed_extensions = ("{" + ", ".join(f'"{extension.strip()}"'
                                                                     for extension in ext.split(","))
                                      + "}")
                script = (f'POSIX path of (choose file with prompt "{prompt}" '
                          f'of type {allowed_extensions})')
            else:
                script = f'POSIX path of (choose file with prompt "{prompt}")'

        try:
            result = _subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                return {"path": None, "cancelled": True}
            # osascript returns the path with a trailing newline
            chosen = result.stdout.strip().rstrip("/")
            return {"path": chosen}
        except _subprocess.TimeoutExpired:
            return {"path": None, "cancelled": True}
        except FileNotFoundError:
            return {"path": None, "error": "osascript not found — macOS only"}
        except Exception as error:
            return {"path": None, "error": str(error)}

    return await asyncio.to_thread(_open_dialog)
