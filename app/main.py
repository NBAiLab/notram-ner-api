import json
import os
from typing import List

from celery.result import AsyncResult
from fastapi import FastAPI

from app.schemas import NerTextRequest, NerUrnRequest
from app.tasks import app as task_app
from app.tasks import run_model_task
from app.util import urn_to_path

URN_BASE_PATH = os.environ.get("URN_BASE_PATH", "app/urn")

description = """
API for communication with Named Entity Recognition (NER) model based on NoTraM (Norwegian Transformer Model).
"""

app = FastAPI(
    title="notram-ner-api",
    description=description,
    version="0.1.0",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


@app.get("/entities/groups", response_model=List[str])
async def groups():
    """
    Get available entity groups.
    """
    from tasks import model
    entity_groups = []
    for label in model.config.id2label.values():
        s = label.split("-")
        if s[0] != "I":
            entity_groups.append(s[-1])
    return entity_groups


@app.post("/entities/text")
async def named_entities_from_text(body: NerTextRequest):
    """
    Get named entities for a specific text.
    """

    res: AsyncResult = run_model_task.delay(
        text=body.text,
        include_entities=body.include_entities,
        do_group_entities=body.group_entities
    )

    if body.wait:
        res.get()

    return {"status": res.status, "uuid": res.id, "result": res.result}


@app.post("/entities/urn")
async def named_entities_from_urn(body: NerUrnRequest):
    """
    Get named entities for a specific URN.
    """
    path = os.path.join(URN_BASE_PATH, urn_to_path(body.urn))
    with open(path) as file:
        jsonl = [json.loads(line) for line in file]

    jsonl = jsonl[0]  # Assuming only one record for now
    all_text = "\n".join([paragraph["text"] for paragraph in jsonl["paragraphs"]])
    # TODO keep track of index?

    return await named_entities_from_text(
        NerTextRequest(
            text=all_text,
            include_entities=body.include_entities,
            do_group_entities=body.group_entities,
            wait=body.wait
        )
    )


@app.get("/task/{uuid}")
async def task_result(uuid: str):
    res = AsyncResult(uuid, app=task_app)
    return {"status": res.status, "uuid": res.id, "result": res.result}
