"""The classic CSI-sensing-literature recognition backend (WiFiSenseSurvey CSUR'19):
`StandardScaler` + a calibrated SVC, kept for A/B against 'mlp' on real recordings.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from wavetrace.adapters.recognition._sklearn_pipeline import SklearnPipelineBackend


class SvmBackend(SklearnPipelineBackend):
    """`StandardScaler` + calibrated SVC. `predict_proba` = Platt/sigmoid scaling."""

    def _build_classifier(self, config) -> CalibratedClassifierCV:
        # sklearn 1.9 deprecated SVC(probability=True); this is the documented replacement.
        return CalibratedClassifierCV(SVC(random_state=config.seed), ensemble=False)
