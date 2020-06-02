# CVD
Cardio Vascular Disease Detection Ineuron Hackathon


Setup Instructions 

#Set PYTHONPATH
export PYTHONPATH=currentDirectory/cvd_model --> parent directory,there is another directory cvd_model inside this

#To train pipeline
run python train_pipeline

#For unit test
Run tox -r 

#To create module 
run python setup.py sdist bdist_wheel
run pip install -e cvd_model/

#ml_api
go to docker folder
run sudo docker-compose -f docker-compose-yml up -d --build

grafana endpoint:
localhost:3000

swagger: localhost:5000/ui

prometheus: localhost:9090
