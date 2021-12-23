import os
import re
from distutils.util import strtobool
from typing import Optional, List

from bs4 import BeautifulSoup
from sentence_splitter import SentenceSplitter
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, NerPipeline
from transformers.pipelines import AggregationStrategy

# Some environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
USE_QUEUE = bool(strtobool(os.environ.get("ENABLE_TASK_QUEUE", "False")))
USE_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN", "false")
if USE_AUTH_TOKEN.lower() in ("true", "false"):
    USE_AUTH_TOKEN = bool(strtobool(USE_AUTH_TOKEN))
ROOT_PATH = os.environ.get("ROOT_PATH", "")
URN_BASE_PATH = os.environ.get("URN_BASE_PATH", None)
MODEL_PATH = os.environ.get("MODEL_PATH", "./model")
DO_BATCHING = bool(strtobool(os.environ.get("DO_BATCHING", "False")))
DEVICE = int(os.environ.get("DEVICE", -1))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 512))
VERSION = str(os.environ.get("VERSION", "1"))
SPLIT_LANG = os.environ.get("SPLIT_LANG", "no")

if SPLIT_LANG.lower() == "disable":
    SENTENCE_SPLITTER = None
else:
    SENTENCE_SPLITTER = SentenceSplitter(language=SPLIT_LANG)


def load_model(path):
    if os.path.exists(path) and os.listdir(path):
        config = AutoConfig.from_pretrained(
            os.path.join(path, "config.json"),
        )
        model = AutoModelForTokenClassification.from_pretrained(
            path, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path, config=config, use_fast=True, model_max_length=MAX_LENGTH
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            path, use_auth_token=USE_AUTH_TOKEN
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=True,
            model_max_length=MAX_LENGTH,
            use_auth_token=USE_AUTH_TOKEN
        )

    args = dict(
        model=model,
        tokenizer=tokenizer,
        ignore_labels=["O"],
        aggregation_strategy=AggregationStrategy.AVERAGE,
        # ignore_subwords=True,
        device=DEVICE
    )
    if DO_BATCHING:
        from .custom_ner_pipeline import StridedNerPipeline
        args.update(dict(sentence_splitter=SENTENCE_SPLITTER))
        pipe_class = StridedNerPipeline
    else:
        pipe_class = NerPipeline
    return pipe_class(**args)


def run_model(model: NerPipeline, text: str, sentence_splitter: SentenceSplitter,
              include_entities: Optional[List[str]] = None, do_group_entities: bool = False,
              validate: bool = True):
    # texts = batch_by_sentence(
    #     text=text,
    #     max_len=min(model.tokenizer.model_max_length, 1024),
    #     sentence_splitter=sentence_splitter
    # )
    texts = text
    out = model(texts)

    # Adjust indices up
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

    if include_entities is not None:
        res = [r for r in res if r["entity_group"] in include_entities]

    res = clean_torch_floats(res)

    if validate:
        validate_indices(original_text=text, res=res)

    if do_group_entities:
        res = group_entities(res)

    return res


def batch_by_sentence(text: str, max_len: int, sentence_splitter: SentenceSplitter):
    if len(text) > max_len:
        sentences = sentence_splitter.split(text)
        # return sentences
        texts = [""]
        tot_len = 0
        for sentence in sentences:
            # Since sentence-splitter strips sentences, we must find start by searching from end of last sentence
            try:
                starting_index = text.index(sentence, tot_len)
                sentence = sentence.rjust(len(sentence) + starting_index - tot_len)
            except ValueError:
                pass  # TODO handle? Will break (at least some) returned indices
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
        texts = [text]
    return texts


def group_entities(res):
    print("Grouping...")
    entity_words = {}
    for r in res:
        entity_words.setdefault(r["entity_group"], []).append(r)
    schema_compatible = []
    for key, value in entity_words.items():
        present = {}
        for entity in value:
            if entity["word"] in present:
                orig_occurrence = present[entity["word"]]
                orig_occurrence["scores"].append(entity["score"])
                orig_occurrence["starts"].append(entity["start"])
                orig_occurrence["ends"].append(entity["end"])
            else:
                entity["scores"] = [entity.pop("score")]
                entity["starts"] = [entity.pop("start")]
                entity["ends"] = [entity.pop("end")]
                present[entity["word"]] = entity
        schema_compatible.append({"entity_group": key, "items": list(present.values())})
    res = schema_compatible
    return res


def clean_torch_floats(out):
    # Turn torch objects into floats so they can be sent
    res = []
    for group in out:
        group["score"] = group["score"].item()
        group["start"] = group["start"].item()
        group["end"] = group["end"].item()
        res.append(group)
    return res


def validate_indices(original_text, res):
    for r in res:  # TODO Investigate: 'Geir.Geir' -> 'Geir. Geir' in result somehow (though indices are correct)
        start, end, word = r["start"], r["end"], r["word"]
        if original_text[start: end] != word:
            print(f"Index mismatch, word: '{word}' not at {start}-{end}")


def urn_to_path(urn: str):
    digibok = re.match(r"digibok_(\d{4})(\d{2})(\d{2})\d+\.jsonl", urn)
    if digibok:
        year, month, day = digibok.groups()
        return f"book/text/{year}/{month}/{day}/{urn}"
    newspaper = re.match(r"(\w+)_null_null_(\d{4})(\d{2})(\d{2})_\d+_\d+_\d+\.jsonl", urn)
    if newspaper:
        name, year, month, day = newspaper.groups()
        return f"newspaper/ocr/text/{year}/{month}/{day}/{name}/{urn}"

    return None


def get_text(connection_or_html) -> str:
    """
    Uses BeautifulSoup to get text from HTML
    """
    # https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text/1983219#1983219
    if isinstance(connection_or_html, BeautifulSoup):
        soup = connection_or_html
    else:
        soup = BeautifulSoup(connection_or_html, "html5lib")

    [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
    [s.extract() for s in soup.find_all(attrs={"style": re.compile("display: ?none|visibility: ?hidden")})]

    split = [re.sub(r"\s+", " ", s) for s in soup.stripped_strings]
    split = [s for s in split if s]

    return "\t".join(split)
