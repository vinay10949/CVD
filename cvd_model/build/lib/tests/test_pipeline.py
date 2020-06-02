from cvd_model import pipeline
from cvd_model.config.core import config
from cvd_model.processing.validation import validate_inputs


def BMI(data):
    return data["weight"] / (data["height"] / 100) ** 2


def test_pipeline_drops_unnecessary_features(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    X_train["age"] = X_train["age"] / 365
    X_train["bmi"] = X_train.apply(BMI, axis=1)
    X_transformed, _ = pipeline.cardio_training_pipe._fit(X_train, y_train)
    assert config.model_config.drop_features not in X_train.columns


def test_pipeline_predict_takes_validated_input(pipeline_inputs, sample_input_data):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    X_train["age"] = X_train["age"] / 365
    X_train["bmi"] = X_train.apply(BMI, axis=1)
    pipeline.cardio_training_pipe.fit(X_train, y_train)

    sample_input_data.drop("cardio", axis=1, inplace=True)
    # When
    validated_inputs, errors = validate_inputs(input_data=sample_input_data[0:2])
    validated_inputs["age"] = sample_input_data["age"] / 365
    validated_inputs["bmi"] = sample_input_data.apply(BMI, axis=1)
    config.model_config.features.append("bmi")
    predictions = pipeline.cardio_training_pipe.predict(
        validated_inputs[config.model_config.features]
    )
    # Then
    assert predictions is not None
    assert errors is None
