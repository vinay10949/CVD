from sklearn.pipeline import Pipeline
from cvd_model.processing import preprocessors as pp
from cvd_model.config.core import config
from cvd_model.config.core import INIT_MODEL
from sklearn.preprocessing import StandardScaler
import joblib

import logging


_logger = logging.getLogger(__name__)
from cvd_model.config.core import INIT_MODEL

catboost_calibrated_model = joblib.load(INIT_MODEL)


cardio_training_pipe = Pipeline(
    [
        (
            "discretize_variables",
            pp.DiscretizeVariable(variables=config.model_config.discretize_variables),
            #  ("ageDiscretizer", age_discretizer_model),
        ),
        ("discretize_bmi", pp.DiscretizeBMI(),),
        ("calculate_blood_pressure_level", pp.CalculateBloodPressureLevel(),),
        (
            "gender_ohe_encoding",
            pp.OHEEncoder(variables=[config.model_config.ohe_feature]),
            #  ("oheGender", ohe_model),
        ),
        (
            "jamestien_encoding",
            pp.JamesStienEncoder(variables=config.model_config.jamesstien_encoding),
            # ("jamesStienModel", jamestienEnc_model),
        ),
        ("scaler", StandardScaler()),
        ("Cat Boost Calibrate", catboost_calibrated_model),
    ]
)
