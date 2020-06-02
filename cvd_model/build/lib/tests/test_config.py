from pathlib import Path

from cvd_model.config.core import (
    create_and_validate_config,
    fetch_config_from_yaml,
)

import pytest
from pydantic import ValidationError


TEST_CONFIG_TEXT = """
package_name: cvd_model
training_data_file: train.csv
test_data_file: test.csv
drop_features: id
pipeline_name: catboost
pipeline_save_file: catboost_regression_output_v
target: cardio
test_size: 0.3
features:
  - age
  - gender
  - height
  - weight
  - active
  - gluc
  - alco
  - ap_hi
  - ap_lo
  - cholesterol
  - smoke
  - cardio

numerical_vars:
  - age
  - gender
  - height
  - weight
  - active
  - gluc
  - alco
  - ap_hi
  - ap_lo
  - cholesterol
  - smoke
  - cardio
 
numerical_na_not_allowed:
  - age
  - gender
  - height
  - weight
  - active
  - gluc
  - alco
  - ap_hi
  - ap_lo
  - cholesterol
  - smoke
  - cardio
    
random_state: 0

ohe_feature: gender

jamesstien_encoding:
  - bmi_category
  - blood_pressure_level
  - age

discretize_variables: age
"""

INVALID_TEST_CONFIG_TEXT = """
training_data_file: train.csv
test_data_file: test.csv
drop_features: id
pipeline_name: catboost
pipeline_save_file: catboost_regression_output_v
target: cardio
test_size: 0.1
features:
  - age
  - gender
  - height
  - weight
  - active
  - gluc
  - alco
  - ap_hi
  - ap_lo
  - cholesterol
  - smoke

numerical_vars:
  - age
  - gender
  - height
  - weight
  - active
  - gluc
  - alco
  - ap_hi
  - ap_lo
  - cholesterol
  - smoke
  - cardio
  

numerical_na_not_allowed:
  - age
  - gender
  - height
  - weight
  - active
  - gluc
  - alco
  - ap_hi
  - ap_lo
  - cholesterol
  - smoke
  
random_state: 0

ohe_feature: gender

jamesstien_encoding:
  - bmi_category
  - blood_pressure_level
  - age

discretize_variables: age
"""


def test_fetch_config_structure(tmpdir):
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"
    config_1.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    # When
    config = create_and_validate_config(parsed_config=parsed_config)
    # Then
    assert config.model_config
    assert config.app_config


def test_config_validation_raises_error_for_invalid_config(tmpdir):
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"

    config_1.write_text(INVALID_TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)
    # When
    with pytest.raises(ValidationError) as excinfo:
        a = create_and_validate_config(parsed_config=parsed_config)
    # Then
    # assert "not in the allowed set" in str(excinfo.value)


def test_missing_config_field_raises_validation_error(tmpdir):
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"
    TEST_CONFIG_TEXT = """package_name: cvd_model"""
    config_1.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    # When
    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert "field required" in str(excinfo.value)
    assert "pipeline_name" in str(excinfo.value)
