from cvd_model.predict import make_prediction
from cvd_model.config.core import config
from sklearn.metrics import fbeta_score
from cvd_model.processing.validation import validate_inputs


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in pos_probs]


def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype("int")


def BMI(data):
    return data["weight"] / (data["height"] / 100) ** 2


def test_prediction_quality_against_benchmark(raw_training_data, sample_input_data):
    # Given
    input_df = sample_input_data.drop(config.model_config.target, axis=1)
    output_df = sample_input_data[config.model_config.target]
    # input_df, errors = validate_inputs(input_data=input_df)
    # Generate rough benchmarks (you would tweak depending on your model)
    benchmark_flexibility = 0.50
    # setting ndigits to -4 will round the value to the nearest 10,000 i.e. 210,000
    benchmark_lower_boundary = 0.40
    benchmark_upper_boundary = 1.0
    subject = make_prediction(input_data=input_df[0:1000])
    # Then
    assert subject is not None
    prediction = subject.get("predictions")
    score = fbeta_score(output_df[0:1000], to_labels(prediction, 0.30), beta=3)
    print("FBeta score ", score)
    assert isinstance(score, float)
    assert score > benchmark_lower_boundary
    assert score < benchmark_upper_boundary


# For SHadow Model or New model for A/B Testing
def test_prediction_quality_against_another_model(raw_training_data, sample_input_data):
    # Given
    input_df = raw_training_data.drop(config.model_config.target, axis=1)
    output_df = raw_training_data[config.model_config.target]
    current_predictions = make_prediction(input_data=input_df[0:5000]).get(
        "predictions"
    )

    # alternative_predictions = alt_make_prediction(input_data=input_df[0:5000])

    # When
    current_score = fbeta_score(
        output_df[0:5000], to_labels(current_predictions, 0.30), beta=3
    )

    alternate_score = fbeta_score(
        output_df[0:5000], to_labels(current_predictions, 0.30), beta=3
    )

    # Then
    assert current_score == alternate_score
