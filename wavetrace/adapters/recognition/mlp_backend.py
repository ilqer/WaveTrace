"""The default recognition backend: `StandardScaler` + a single-hidden-layer `MLPClassifier`.

Native `predict_proba` (the soft segment vote needs calibrated probabilities) and its weight
matrices port directly to a numpy-only forward pass for an ESP32-only deployment.
"""

from sklearn.neural_network import MLPClassifier

from wavetrace.adapters.recognition._sklearn_pipeline import SklearnPipelineBackend


class MlpBackend(SklearnPipelineBackend):
    """`StandardScaler` + `MLPClassifier` — the user-locked default (`ModelConfig.backend`)."""

    def _build_classifier(self, config) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=(config.hidden,),
            max_iter=3000,
            random_state=config.seed,
        )
