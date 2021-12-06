from typing import Union, List, Optional

import numpy as np
import torch
from sentence_splitter import SentenceSplitter
from transformers import TokenClassificationPipeline, PreTrainedModel, TFPreTrainedModel, PreTrainedTokenizer
from app.util import batch_by_sentence


def _softmax(logits):
    ex = np.exp(logits - logits.max(-1, keepdims=True))
    return ex / ex.sum(-1, keepdims=True)


class StridedNerPipeline(TokenClassificationPipeline):
    def __init__(self, model: Union["PreTrainedModel", "TFPreTrainedModel"], tokenizer: PreTrainedTokenizer,
                 batch_size=8, max_len=None, strides_per_token=2, sentence_splitter: Optional[SentenceSplitter] = None,
                 *args, **kwargs):
        super().__init__(model, tokenizer, *args, **kwargs)
        self.batch_size = batch_size
        self.max_len = model.config.max_position_embeddings if max_len is None else max_len
        assert self.max_len % strides_per_token == 0
        self.strides_per_token = strides_per_token
        self.stride_len = self.max_len // self.strides_per_token
        self.sentence_splitter = sentence_splitter
        if self.sentence_splitter is not None:
            self.strides_per_token = 1
            self.stride_len = self.max_len

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        _inputs, offset_mappings = self._args_parser(inputs, **kwargs)

        answers = []
        for i, text in enumerate(_inputs):
            if self.sentence_splitter is not None:  # TODO
                sentences = batch_by_sentence(text, self.max_len, self.sentence_splitter)
                tokens = self.tokenizer(
                    sentences,
                    return_attention_mask=False,
                    return_tensors=self.framework,
                    padding=True,
                    truncation=True,
                    return_special_tokens_mask=True,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=self.tokenizer.is_fast,
                    pad_to_multiple_of=self.max_len,
                )
            else:
                tokens = self.tokenizer(
                    text,
                    return_attention_mask=False,
                    return_tensors=self.framework,
                    padding=True,
                    truncation=True,
                    return_special_tokens_mask=True,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=self.tokenizer.is_fast,
                    # pad_to_multiple_of=max_len,
                    stride=self.stride_len % self.max_len
                )
            tokens.pop("overflow_to_sample_mapping")

            if self.tokenizer.is_fast:
                offset_mapping = tokens.pop("offset_mapping").cpu().numpy()
            elif offset_mappings:
                offset_mapping = offset_mappings[i]
            else:
                offset_mapping = None

            special_tokens_mask = tokens.pop("special_tokens_mask").cpu().numpy()

            # Special token mask and offset mapping need to be flattened for use in gather method
            leftover_strides = (len(special_tokens_mask) - 1) % self.strides_per_token
            special_tokens_mask = np.concatenate(
                [special_tokens_mask[::self.strides_per_token].flatten()]
                + ([] if leftover_strides == 0
                   else [special_tokens_mask[-1][-leftover_strides * self.stride_len:].flatten()])
            )
            if offset_mapping is not None:
                offset_mapping = np.concatenate(
                    [offset_mapping[::self.strides_per_token].reshape(-1, 2)]
                    + ([] if leftover_strides == 0
                       else [offset_mapping[-1][-leftover_strides * self.stride_len:].reshape(-1, 2)])
                )

            scores = np.zeros((special_tokens_mask.shape[0], self.model.config.num_labels))
            counts = np.zeros(scores.shape)
            all_input_ids = np.zeros(special_tokens_mask.shape[0], dtype=np.int32)
            n_samples = tokens["input_ids"].shape[0]

            # Forward
            tokens = self.ensure_tensor_on_device(**tokens)

            start_offset = 0
            for index in range(0, n_samples, self.batch_size):
                batch = {
                    name: tensor[index:index + self.batch_size] if isinstance(tensor, torch.Tensor) else tensor
                    for name, tensor in tokens.items()
                }

                if self.framework == "tf":
                    entities = self.model(batch)[0].numpy()  # Not tested
                    batch_input_ids = batch["input_ids"].numpy()
                else:
                    with torch.no_grad():
                        entities = self.model(**batch)[0].cpu().numpy()
                        batch_input_ids = batch["input_ids"].cpu().numpy()

                size = entities.shape[1]
                for s in range(0, entities.shape[0]):
                    scores[start_offset: start_offset + size] += entities[s]
                    counts[start_offset: start_offset + size] += 1

                    all_input_ids[start_offset: start_offset + size] = batch_input_ids[s]
                    start_offset += self.stride_len

            scores = np.divide(scores, counts)

            scores = _softmax(scores)
            pre_entities = self.gather_pre_entities(text, all_input_ids, scores, offset_mapping, special_tokens_mask)
            grouped_entities = self.aggregate(pre_entities, self.aggregation_strategy)
            # Filter anything that is in self.ignore_labels
            entities = [
                entity
                for entity in grouped_entities
                if (entity.get("entity", None) not in self.ignore_labels
                    and entity.get("entity_group", None) not in self.ignore_labels)
            ]
            answers.append(entities)

        if len(answers) == 1:
            return answers[0]
        return answers
        # self.model()

        # _inputs, offset_mappings = self._args_parser(inputs, **kwargs)
        #
        # answers = []
        #
        # for i, sentence in enumerate(_inputs):
        #
        #     # Manage correct placement of the tensors
        #     with self.device_placement():
        #
        #         tokens = self.tokenizer(
        #             sentence,
        #             return_attention_mask=False,
        #             return_tensors=self.framework,
        #             truncation=True,
        #             return_special_tokens_mask=True,
        #             return_offsets_mapping=self.tokenizer.is_fast,
        #         )
        #         if self.tokenizer.is_fast:
        #             offset_mapping = tokens.pop("offset_mapping").cpu().numpy()[0]
        #         elif offset_mappings:
        #             offset_mapping = offset_mappings[i]
        #         else:
        #             offset_mapping = None
        #
        #         special_tokens_mask = tokens.pop("special_tokens_mask").cpu().numpy()[0]
        #
        #         # Forward
        #         if self.framework == "tf":
        #             entities = self.model(tokens.data)[0][0].numpy()
        #             input_ids = tokens["input_ids"].numpy()[0]
        #         else:
        #             with torch.no_grad():
        #                 tokens = self.ensure_tensor_on_device(**tokens)
        #                 entities = self.model(**tokens)[0][0].cpu().numpy()
        #                 input_ids = tokens["input_ids"].cpu().numpy()[0]
        #
        #     scores = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
        #     pre_entities = self.gather_pre_entities(sentence, input_ids, scores, offset_mapping, special_tokens_mask)
        #     grouped_entities = self.aggregate(pre_entities, self.aggregation_strategy)
        #     # Filter anything that is in self.ignore_labels
        #     entities = [
        #         entity
        #         for entity in grouped_entities
        #         if entity.get("entity", None) not in self.ignore_labels
        #         and entity.get("entity_group", None) not in self.ignore_labels
        #     ]
        #     answers.append(entities)
        #
        # if len(answers) == 1:
        #     return answers[0]
        # return answers
