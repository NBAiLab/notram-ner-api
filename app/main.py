from pathlib import Path

import torch
from fastapi import FastAPI
from transformers import BertForTokenClassification, BertTokenizer, pipeline

model: BertForTokenClassification = BertForTokenClassification.from_pretrained("notram-eval4-norne-bokmaal/output")
tokenizer = BertTokenizer.from_pretrained("notram-eval4-norne-bokmaal/output")

label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-GPE_LOC", "I-GPE_LOC", "B-PROD", "I-PROD", "B-LOC", "I-LOC",
              "B-GPE_ORG", "I-GPE_ORG", "B-DRV", "I-DRV", "B-EVT", "I-EVT", "B-MISC", "I-MISC"]
model.config.id2label = {i: l for i, l in enumerate(label_list)}
model.config.label2id = {l: i for i, l in enumerate(label_list)}
pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer)

app = FastAPI()


@app.get("/named_entities")
def get_named_entities():
    test_str = "Heisann! Dette er en test av Nasjonalbibliotekets NER-modell."
    res = pipeline(test_str)
    res = [{"entity": r["entity"], "score": r["score"].item(), "word": r["word"]} for r in res]
    return res
