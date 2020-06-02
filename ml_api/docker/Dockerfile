FROM python:3.7.5-slim-buster
RUN mkdir -p /opt/app
COPY requirements /opt/app/requirements
COPY cvd_module /opt/app/cvd_module

RUN pip install --upgrade pip

RUN python3 -m pip install -r /opt/app/requirements/requirements.txt
RUN pip install /opt/app/cvd_module/*

# ensure we can run the make commands
RUN apt-get update -y && \
 	apt-get install -y make && \
 	apt-get install -y libffi-dev gcc && \
 	# for swagger
 	apt-get install -y curl && \
 	# for postgres driver
 	apt-get install -y libpq-dev


ENV PYTHONPATH "${PYTHONPATH}:/opt/app/"

ADD . /opt/app
WORKDIR /opt/app
