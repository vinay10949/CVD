import json
import time

import numpy as np
import pytest

from api.persistence.models import (
    GradientBoostingModelPredictions,
    CatBoostCalibratedModelPredictions,
)
import pandas as pd


@pytest.mark.integration
def test_health_endpoint(client):
    # When
    response = client.get("/")

    # Then
    assert response.status_code == 200
    assert json.loads(response.data) == {"status": "ok"}


@pytest.mark.integration
@pytest.mark.parametrize(
    "api_endpoint, expected_no_predictions",
    (
        (
            "v1/predictions/predict",
            # test csv contains 1459 rows
            # we expect 2 rows to be filtered
            1451,
        ),
        (
            "v1/predictions/gradient",
            # we expect 8 rows to be filtered
            1457,
        ),
    ),
)
def test_prediction_endpoint(
    api_endpoint, expected_no_predictions, client, test_inputs_df
):
    # Given
    # Load the test dataset which is included in the model package
    test_inputs_df = pd.read_csv("test.csv")  # dataframe
    response = client.post(api_endpoint, json=test_inputs_df.to_dict(orient="records"))

    # Then
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["errors"] is None
    assert len(data["predictions"]) == expected_no_predictions


@pytest.mark.integration
def test_prediction_validation(
    field, field_value, index, expected_error, client, test_inputs_df
):
    # Given
    # Check gradient_boosting_model.processing.validation import PersonDataInputSchema
    # and you will see the expected values for the inputs to the Cardio vascular Diseasel   prediction
    # model. In this test, inputs are changed to incorrect values to check the validation.
    test_inputs_df.loc[index, field] = field_value

    # When
    response = client.post(
        "/v1/predictions/gradient", json=test_inputs_df.to_dict(orient="records")
    )

    # Then
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data == expected_error


@pytest.mark.integration
def test_prediction_data_saved(client, app, test_inputs_df):
    # Given
    initial_gradient_count = app.db_session.query(
        GradientBoostingModelPredictions
    ).count()
    initial_lasso_count = app.db_session.query(CatBoostCalibratedModelPredictions).count()

    # When
    response = client.post(
        "/v1/predictions/predict", json=test_inputs_df.to_dict(orient="records")
    )

    # Then
    assert response.status_code == 200
    assert (
        app.db_session.query(CatBoostCalibratedModelPredictions).count() == initial_lasso_count + 1
    )

    # The gradient prediction save occurs on a separate async thread which can take
    # time to complete. We pause the test briefly to allow the save operation to finish.
    time.sleep(2)
    assert (
        app.db_session.query(GradientBoostingModelPredictions).count()
        == initial_gradient_count + 1
    )
