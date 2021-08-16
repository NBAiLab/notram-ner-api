from celery import Celery

from app.util import run_model, load_model, MODEL_PATH, SENTENCE_SPLITTER

app = Celery(
    name='tasks',
    broker='redis://localhost:6379',
    backend='db+sqlite:///db.sqlite3'
)

model = load_model(MODEL_PATH)


@app.task(name="run_model_task")
def run_model_task(*args, **kwargs):
    return run_model(model=model, sentence_splitter=SENTENCE_SPLITTER, *args, **kwargs)
