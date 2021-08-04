import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    NerPipeline, PreTrainedTokenizerFast,
)
from transformers.pipelines import AggregationStrategy

# from ts.metrics.dimension import Dimension
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "notram-eval4-norne-bokmaal/output")  # TODO better way?
LABELS = os.path.join(MODEL_PATH, "labels.txt")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_labels():
    if os.path.isfile(LABELS):  # TODO add labels file?
        with open(LABELS, "r") as f:
            labels = [line.strip() for line in f.read().strip().splitlines()]
        if "O" not in labels:
            labels = ["O"] + labels
    else:
        labels = [
            "O",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-GPE_LOC",
            "I-GPE_LOC",
            "B-PROD",
            "I-PROD",
            "B-LOC",
            "I-LOC",
            "B-GPE_ORG",
            "I-GPE_ORG",
            "B-DRV",
            "I-DRV",
            "B-EVT",
            "I-EVT",
            "B-MISC",
            "I-MISC",
        ]
    return labels


def load_model():
    labels = get_labels()
    label_map = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)
    config = AutoConfig.from_pretrained(
        os.path.join(MODEL_PATH, "config.json"),
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
    )
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_PATH, config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, model_max_length=512,
                                              stride=256, return_overflowing_tokens=True)

    return NerPipeline(
        model=model,
        tokenizer=tokenizer,
        ignore_labels=["O"],
        aggregation_strategy=AggregationStrategy.SIMPLE,

        # ignore_subwords=True,
    )


model = load_model()

app = FastAPI()


def _clean(out):
    # Turn torch objects into floats so they can be sent
    res = []
    for group in out:
        group["score"] = group["score"].item()
        group["start"] = group["start"].item()
        group["end"] = group["end"].item()
        res.append(group)
    return res


class TextBody(BaseModel):  # Need this to receive json data
    text: str


@app.post("/named_entities")  # TODO Post or get? JSON or url encoded
async def get_named_entities(body: TextBody):
    # TODO group by label? Deal with max length (window+stride)?
    text = body.text
    out = model(text)
    return _clean(out)
