from sklearn.model_selection import train_test_split

import pipeline
from cvd_model.processing.data_management import (
    load_dataset,
    save_pipeline,
)
from cvd_model.config.core import config
from cvd_model import __version__ as _version

import logging


_logger = logging.getLogger(__name__)


def BMI(data):
    return data["weight"] / (data["height"] / 100) ** 2


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)
    # Convert days in age to years
    data["age"] = data["age"] / 365

    # Drop Duplicates
    data = data.drop_duplicates(keep="first")

    # calculate bmi from height and weight
    data["bmi"] = data.apply(BMI, axis=1)

    # drop outliers
    data.drop(
        data[
            (data["ap_hi"] > data["ap_hi"].quantile(0.975))
            | (data["ap_hi"] < data["ap_hi"].quantile(0.025))
        ].index,
        inplace=True,
    )
    data.drop(
        data[
            (data["ap_lo"] > data["ap_lo"].quantile(0.975))
            | (data["ap_lo"] < data["ap_lo"].quantile(0.025))
        ].index,
        inplace=True,
    )

    data.drop(
        data[
            (data["height"] > data["height"].quantile(0.999))
            | (data["height"] < data["height"].quantile(0.025))
        ].index,
        inplace=True,
    )
    data.drop(
        data[
            (data["weight"] > data["weight"].quantile(0.999))
            | (data["weight"] < data["weight"].quantile(0.025))
        ].index,
        inplace=True,
    )
    config.model_config.features.append("bmi")

    print(config.model_config.features)
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size=0.1,
        random_state=0,
    )  # we are setting the seed here
    print(pipeline.cardio_training_pipe)
    pipeline.cardio_training_pipe.fit(X_train[config.model_config.features], y_train)
    _logger.info(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=pipeline.cardio_training_pipe)


if __name__ == "__main__":
    run_training()
