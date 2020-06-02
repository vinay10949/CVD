import typing as t

from cvd_model.config.core import config

import numpy as np
import pandas as pd
from marshmallow import fields, Schema, ValidationError


class CVDDataInputSchema(Schema):
    age = fields.Integer()
    gender = fields.Integer()
    height = fields.Integer()
    weight = fields.Float()
    active = fields.Integer()
    gluc = fields.Integer()
    alco = fields.Integer()
    ap_hi = fields.Integer()
    ap_lo = fields.Integer()
    cholesterol = fields.Integer()
    smoke = fields.Integer()


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    if input_data[config.model_config.numerical_na_not_allowed].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.model_config.numerical_na_not_allowed
        )
    return validated_data


def validate_inputs(
    *, input_data: pd.DataFrame
) -> t.Tuple[pd.DataFrame, t.Optional[dict]]:
    """Check model inputs for unprocessable values."""
    validated_data = drop_na_inputs(input_data=input_data)
    # set many=True to allow passing in a list
    schema = CVDDataInputSchema(many=True)
    errors = None
    try:
        # replace numpy nans so that Marshmallow can validate
        schema.load(validated_data.replace({np.nan: None}).to_dict(orient="records"))
    except ValidationError as exc:
        print("ERROR ", exc.messages)
        errors = exc.messages
    return validated_data, errors
