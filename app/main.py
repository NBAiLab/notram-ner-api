import asyncio
import json
import logging
import os
import re
import time
from contextlib import AbstractContextManager

from fastapi import FastAPI
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    NerPipeline
)
from transformers.pipelines import AggregationStrategy
from sentence_splitter import SentenceSplitter

from app.schemas import NerTextRequest, NerResponse, NerUrnRequest

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "app/model")  # TODO better way?
LABELS = os.path.join(MODEL_PATH, "labels.txt")
URN_BASE_PATH = os.environ.get("URN_BASE_PATH", "app/urn")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sentence_splitter = SentenceSplitter(language="no")


def load_model():
    config = AutoConfig.from_pretrained(
        os.path.join(MODEL_PATH, "config.json"),
    )
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_PATH, config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, model_max_length=512)

    return NerPipeline(
        model=model,
        tokenizer=tokenizer,
        ignore_labels=["O"],
        aggregation_strategy=AggregationStrategy.AVERAGE,
        # ignore_subwords=True,
    )


model = load_model()

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
    }
)


def _clean(out):
    # Turn torch objects into floats so they can be sent
    res = []
    for group in out:
        group["score"] = group["score"].item()
        group["start"] = group["start"].item()
        group["end"] = group["end"].item()
        res.append(group)
    return res


# def _split_sentences(txt):  # Spacy is weird about removing characters so this is used instead
#     punct_chars = ''.join(Sentencizer.default_punct_chars)
#     return re.split(rf"(?<=[^A-Z][a-z][{punct_chars}])(?=\s+[A-Z])|(?=[\n\t])", txt)


@app.post("/entities/text", response_model=NerResponse)
async def named_entities_from_text(body: NerTextRequest) -> NerResponse:
    """
    Get named entities for a specific text.
    """
    max_len = min(model.tokenizer.model_max_length, 1024)
    if len(body.text) > max_len:
        sentences = sentence_splitter.split(body.text)
        texts = [""]
        tot_len = 0
        for sentence in sentences:
            # Since sentence-splitter strips sentences, we must find start by searching from end of last sentence
            starting_index = body.text.index(sentence, tot_len)
            sentence = sentence.rjust(len(sentence) + starting_index - tot_len)
            tot_len += len(sentence)

            if len(texts[-1]) + len(sentence) < max_len:
                texts[-1] += sentence
            elif len(sentence) > max_len:
                words = re.split(r"(?<=\s)", sentence)  # Split without removing any characters
                for word in words:
                    if len(texts[-1]) + len(word) < max_len:
                        texts[-1] += word
                    elif len(word) > max_len:
                        n_steps = len(word) // max_len + 1
                        step_size = len(word) // n_steps + 1
                        for i in range(0, len(word), step_size):
                            texts.append(word[i:i + max_len])  # In the absolute worst case, split up word
                    else:
                        texts.append(word)
            else:
                texts.append(sentence)
    else:
        texts = [body.text]
    # print("\n".join(texts))
    # print([len(model.tokenizer.tokenize(t)) for t in texts])
    out = model(texts)

    if len(out) > 0 and isinstance(out[0], dict):
        res = out
    else:
        res = []
        tot_index = 0
        for text, o in zip(texts, out):
            for r in o:
                r["start"] += tot_index  # Make sure index is consistent
                r["end"] += tot_index
                res.append(r)
            tot_index += len(text)

    if body.include_entities is not None:
        res = [r for r in res if r["entity_group"] in body.include_entities]

    res = _clean(res)

    if body.group_entities:
        entity_words = {}
        for r in res:
            entity_words.setdefault(r["entity_group"], []).append(r)
        schema_compatible = []
        for key, value in entity_words.items():
            schema_compatible.append({"entity_group": key, "items": value})
        res = schema_compatible

    for r in res:  # TODO Investigate: 'Geir.Geir' -> 'Geir. Geir' in result somehow (though indices are correct)
        start, end, word = r["start"], r["end"], r["word"]
        if body.text[start: end] != word:
            print(f"Index mismatch, word: '{word}' not at {start}-{end}")

    return res


@app.get("/entities/groups")
async def groups():
    """
    Get available entity groups.
    """
    return list(model.model.config.id2label.values())


@app.post("/entities/urn", response_model=NerResponse)
async def named_entities_from_urn(body: NerUrnRequest) -> NerResponse:
    """
    Get named entities for a specific URN.
    """
    path = os.path.join(URN_BASE_PATH, _convert_urn(body.urn))
    with open(path) as file:
        jsonl = [json.loads(line) for line in file]

    jsonl = jsonl[0]  # Assuming only one record for now
    # futures = []
    all_text = "\n".join([paragraph["text"] for paragraph in jsonl["paragraphs"]])
    # for paragraph in jsonl["paragraphs"]:
    #     futures.append(get_named_entities_str(NerBody(text=paragraph["text"])))  # Preliminary solution

    return await named_entities_from_text(NerTextRequest(text=all_text,
                                                         include_entities=NerUrnRequest.include_entities,
                                                         group_entities=NerUrnRequest.group_entities))
    # results = await asyncio.gather(*futures)
    # return sum(results, start=[])


def _convert_urn(urn: str):
    digibok = re.match(r"digibok_(\d{4})(\d{2})(\d{2})\d+\.jsonl", urn)
    if digibok:
        year, month, day = digibok.groups()
        return f"book/text/{year}/{month}/{day}/{urn}"
    newspaper = re.match(r"(\w+)_null_null_(\d{4})(\d{2})(\d{2})_\d+_\d+_\d+\.jsonl", urn)
    if newspaper:
        name, year, month, day = newspaper.groups()
        return f"newspaper/ocr/text/{year}/{month}/{day}/{name}/{urn}"

    return None
