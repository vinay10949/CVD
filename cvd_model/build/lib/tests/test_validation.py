from cvd_model.processing.validation import validate_inputs


data_schema = {
    "age": {"range": {"min": 10798, "max": 23713}, "dtype": int,},
    "height": {"range": {"min": 55, "max": 250}, "dtype": int,},
    "weight": {"range": {"min": 11, "max": 200}, "dtype": int,},
    "ap_hi": {"range": {"min": 100, "max": 170}, "dtype": int,},
    "ap_lo": {"range": {"min": 60, "max": 100}, "dtype": int,},
    "smoke": {"range": {"min": 0, "max": 1}, "dtype": int,},
    "active": {"range": {"min": 0, "max": 1}, "dtype": int,},
    "alco": {"range": {"min": 0, "max": 1}, "dtype": int,},
    "cholesterol": {"range": {"min": 0, "max": 3}, "dtype": int,},
    "gluc": {"range": {"min": 0, "max": 3}, "dtype": int,},
    "gender": {"range": {"min": 1, "max": 2}, "dtype": int,},
}


def test_validate_inputs(sample_input_data):
    # When
    sample_input_data = sample_input_data.copy().iloc[[0, 1, 2, 3], :]
    sample_input_data.drop("cardio", axis=1, inplace=True)
    validated_inputs, errors = validate_inputs(input_data=sample_input_data)
    # Then
    assert not errors
    assert len(validated_inputs) == len(sample_input_data)


def test_validate_inputs_identifies_errors(sample_input_data):
    # Given
    # test_inputs = sample_input_data.copy().iloc[[0,1,2,3],:]
    test_inputs = sample_input_data.copy()
    test_inputs.drop("cardio", axis=1, inplace=True)
    # introduce errors
    test_inputs.at[0, "height"] = 100.5678  # we expect a int not float
    # When
    validated_inputs, errors = validate_inputs(input_data=test_inputs)
    # Then
    for feature in validated_inputs.columns:
        if feature == "cardio":
            continue
        assert validated_inputs[feature].max() <= data_schema[feature]["range"]["max"]
        assert validated_inputs[feature].min() >= data_schema[feature]["range"]["min"]
