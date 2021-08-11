import os
import time

from celery import Celery
from sentence_splitter import SentenceSplitter

from app.util import run_model, load_model

MODEL_PATH = os.environ.get("MODEL_PATH", "app/model")  # TODO better way?

app = Celery(
    name='tasks',
    broker='redis://redis:6379',
    backend='db+sqlite:///db.sqlite3'
)

model = load_model(MODEL_PATH)
sentence_splitter = SentenceSplitter(language="no")


@app.task(name="run_model_task")
def run_model_task(*args, **kwargs):
    return run_model(model=model, sentence_splitter=sentence_splitter, *args, **kwargs)
