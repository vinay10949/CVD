import logging
import typing as t

import pandas as pd
from cvd_model import __version__ as _version
from cvd_model.config.core import config
from cvd_model.processing.data_management import load_pipeline
from cvd_model.processing.validation import validate_inputs
import joblib
from cvd_model import PRETRAINED_MODELS

_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_cardio_pipe = load_pipeline(file_name=pipeline_file_name)

ohe_enc = joblib.load(PRETRAINED_MODELS / "ohe.pkl")
ageDiscretizer = joblib.load(PRETRAINED_MODELS / "ageDiscretizer.pkl")
jamesstien_enc = joblib.load(PRETRAINED_MODELS / "jamesStienenc.pkl")


def BMI(data):
    return data["weight"] / (data["height"] / 100) ** 2


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict],) -> dict:
    """Make a prediction using a saved model pipeline."""
    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        X = validated_data[config.model_config.features]
        X["age"] = X["age"] / 365
        X = ageDiscretizer.transform(X)
        X["bmi"] = validated_data.apply(BMI, axis=1)
        # calculate bmi from height and weight
        predictions = _cardio_pipe.predict_proba(X)[:, 1]
        _logger.info(
            f"Making predictions with model version: {_version} "
            f"Predictions: {predictions}"
        )
        results = {"predictions": predictions, "version": _version, "errors": errors}
    return results
