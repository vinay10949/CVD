import json
import logging
import threading

from flask import request, jsonify, Response, current_app
from prometheus_client import Histogram, Gauge, Info,Counter
from cvd_model import __version__ as live_version

from api.config import APP_NAME
from api.persistence.data_access import PredictionPersistence, ModelType
from cvd_model import __version__ as shadow_version
from cvd_model.predict import make_prediction

_logger = logging.getLogger('mlapi')


NO_OF_PATIENTS = Counter(
    name='no_of_patients',
    documentation='Total No of Patients',
    labelnames=['app_name', 'model_name', 'model_version']
)
NO_OF_PATIENTS.labels(
                app_name=APP_NAME,
                model_name=ModelType.CATBOOST.name,
                model_version=live_version)




PREDICTION_TRACKER = Histogram('cardio_vascular_disease_probablities', 'ML Model Prediction on Cardio Vascular Disease')


PREDICTION_TRACKER_PREDICTED_POSITIVE = Gauge(
    name='cardio_vascular_disease_positive',
    documentation='ML Model Prediction For Positive Cardio Diseased Patients',
    labelnames=['app_name', 'model_name', 'model_version']
)

PREDICTION_TRACKER_PREDICTED_POSITIVE.labels(
                app_name=APP_NAME,
                model_name=ModelType.CATBOOST.name,
                model_version=live_version)

PREDICTION_TRACKER_PREDICTED_NEGATIVE = Gauge(
    name='cardio_vascular_disease_negative',
    documentation='ML Model Prediction For Negative Cardio Diseased Patients',
    labelnames=['app_name', 'model_name', 'model_version']
)

PREDICTION_TRACKER_PREDICTED_NEGATIVE.labels(
                app_name=APP_NAME,
                model_name=ModelType.CATBOOST.name,
                model_version=live_version)

MODEL_VERSIONS = Info(
    'model_version_details',
    'Capture model version information',
)

MODEL_VERSIONS.info({
    'live_model': ModelType.CATBOOST.name,
    'live_version': live_version,
    'shadow_model': ModelType.GRADIENT_BOOSTING.name,
    'shadow_version': shadow_version})


def health():
    if request.method == "GET":
        status = {"status": "ok"}
        _logger.debug(status)
        return jsonify(status)


def predict():
    if request.method == "POST":
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()
        for entry in json_data:
            _logger.info(entry)

        # Step 2a: Get and save live model predictions
        print("Input data ",json_data)
        persistence = PredictionPersistence(db_session=current_app.db_session)
        result = persistence.make_save_predictions(
            db_model=ModelType.CATBOOST, input_data=json_data
        )

        # Step 2b: Get and save shadow predictions asynchronously
        if current_app.config.get("SHADOW_MODE_ACTIVE"):
            _logger.debug(
                f"Calling shadow model asynchronously: "
                f"{ModelType.GRADIENT_BOOSTING.value}"
            )
            thread = threading.Thread(
                target=persistence.make_save_predictions,
                kwargs={
                    "db_model": ModelType.GRADIENT_BOOSTING,
                    "input_data": json_data,
                },
            )
            thread.start()

        # Step 3: Handle errors
        if result.errors:
            _logger.warning(f"errors during prediction: {result.errors}")
            return Response(json.dumps(result.errors), status=400)

        # Step 4: Monitoring
        for _prediction in result.predictions:
            PREDICTION_TRACKER.observe(_prediction)
            NO_OF_PATIENTS.labels(
                    app_name=APP_NAME,
                    model_name=ModelType.CATBOOST.name,
                    model_version=live_version).inc(1)
            targetPredicted = 1 if _prediction>0.30 else 0
            if targetPredicted==1:
                PREDICTION_TRACKER_PREDICTED_POSITIVE.labels(
                    app_name=APP_NAME,
                    model_name=ModelType.CATBOOST.name,
                    model_version=live_version).inc(1)
            else: 
                PREDICTION_TRACKER_PREDICTED_NEGATIVE.labels(
                    app_name=APP_NAME,
                    model_name=ModelType.CATBOOST.name,
                    model_version=live_version).inc(1)        
        _logger.info(
            f'Prediction results for model: {ModelType.CATBOOST.name} '
            f'version: {result.model_version} '
            f'Output values: {result.predictions}')

        # Step 5: Prepare prediction response
        return jsonify(
            {
                "predictions": result.predictions,
                "version": result.model_version,
                "errors": result.errors,
            }
        )


def predict_previous():
    if request.method == "POST":
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()

        # Step 2: Access the model prediction function (also validates data)
        result = make_prediction(input_data=json_data)

        # Step 3: Handle errors
        errors = result.get("errors")
        if errors:
            return Response(json.dumps(errors), status=400)

        # Step 4: Split out results
        predictions = result.get("predictions").tolist()
        version = result.get("version")

        # Step 5: Prepare prediction response
        return jsonify(
            {"predictions": predictions, "version": version, "errors": errors}
        )
