#!/bin/bash

set -euox pipefail

MODEL_VERSION="master"
MODEL_VARIANT="candidate"
NUMBER_OF_TESTS="50"


sudo docker ps --all


## Compute the actual predictions (i.e. candidate model)
sudo docker exec --user root differential-tests-actual \
    python3 differential_tests compute sample_payloads differential_tests/actual_results --base-url http://head_ml_api_1:5001

## Copy the actual predictions
sudo docker cp differential-tests-actual:/opt/app/differential_tests/actual_results/. differential_tests/actual_results

echo "===== Running master ... ====="
## Compute the expected marginals (i.e. existing model)
sudo docker exec --user root differential-tests-expected \
    python3 differential_tests compute sample_payloads differential_tests/expected_results --base-url http://master_ml_api_1:5000

## Copy the expected marginals
sudo docker cp differential-tests-expected:/opt/app/differential_tests/expected_results/. differential_tests/expected_results

# then copy all results into the differential-tests-actual container for comparison
sudo docker cp differential_tests/expected_results/. differential-tests-actual:/opt/app/differential_tests/expected_results


## Compare the expected and actual marginals
sudo docker exec differential-tests-actual \
    python3 -m differential_tests compare differential_tests/expected_results differential_tests/actual_results

# clear any sudo docker containers (will stop the script if no containers found)
sudo docker rm $(sudo docker ps -a -q) -f
