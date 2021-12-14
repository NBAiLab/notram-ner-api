# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
#
# RUN apt-get -y update
# RUN apt-get -y upgrade
# RUN apt-get install -y sqlite3 libsqlite3-dev
#
# COPY requirements.txt /app/requirements.txt
# RUN pip install -r /app/requirements.txt
#
# COPY ./app /app/app
# COPY ./model /app/model
#
# # Prestart script runs celery
# COPY prestart.sh /app/prestart.sh


FROM python:3.9

WORKDIR /code

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y sqlite3 libsqlite3-dev

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY ./model /code/model

COPY start.sh /code/start.sh
CMD ["start.sh"]
