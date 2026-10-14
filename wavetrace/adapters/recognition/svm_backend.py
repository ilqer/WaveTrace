from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from wavetrace.adapters.recognition._sklearn_pipeline import SklearnPipelineBackend


class SvmBackend(SklearnPipelineBackend):
    """`predict_proba` is Platt/sigmoid scaling."""

    def _build_classifier(self, config) -> CalibratedClassifierCV:
        # sklearn 1.9 deprecated SVC(probability=True); this is the documented replacement.
        return CalibratedClassifierCV(SVC(random_state=config.seed), ensemble=False)
