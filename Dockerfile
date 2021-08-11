FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y sqlite3 libsqlite3-dev

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY ./app /app/app
# TODO fix celery
#ENTRYPOINT ["celery", "-A", "app.tasks", "worker", "--detach", "-P", "solo"]