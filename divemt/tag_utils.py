"""Utilities for tagging and tokenization."""

import logging
from typing import List, Optional, Tuple, Union

import pandas as pd
import stanza
from tqdm import tqdm

from divemt.cache_utils import CacheDecorator

_STANZA_NLP_MAP = {
    "eng": {"lang": "en", "processors": "tokenize,pos,depparse,ner,lemma"},
    "ara": {"lang": "ar", "processors": "tokenize,pos,depparse,ner,lemma,mwt"},
    "nld": {"lang": "nl", "processors": "tokenize,pos,depparse,ner,lemma"},
    "ita": {"lang": "it", "processors": "tokenize,pos,depparse,ner,lemma,mwt"},
    "tur": {"lang": "tr", "processors": "tokenize,pos,depparse,ner,lemma,mwt"},
    "ukr": {"lang": "uk", "processors": "tokenize,pos,depparse,ner,lemma,mwt"},
    "vie": {"lang": "vi", "processors": "tokenize,pos,depparse,ner,lemma"},
}

stanza.logger.setLevel(logging.WARNING)

_LOADED_NLP = {}

STANZA_FIELDS = {
    "lemma": str,
    "upos": str,
    "feats": str,
    "head": int,
    "deprel": str,
    "start_char": int,
    "end_char": int,
    "ner": str,
}

STANZA_WORD_FIELDS = ["lemma", "upos", "feats", "head", "deprel"]
STANZA_TOKEN_FIELDS = ["start_char", "end_char", "ner"]


def load_nlp(lang: str, tok_only: bool = False):
    if lang not in _STANZA_NLP_MAP:
        try:
            return stanza.Pipeline(lang=lang, processors="tokenize")
        except Exception as e:
            raise ValueError(f"Language {lang} not supported") from e
    if tok_only:
        return stanza.Pipeline(lang=_STANZA_NLP_MAP[lang]["lang"], processors="tokenize")
    return stanza.Pipeline(
        lang=_STANZA_NLP_MAP[lang]["lang"],
        processors=_STANZA_NLP_MAP[lang]["processors"],
    )


def clear_nlp_cache():
    _LOADED_NLP.clear()


def tokenize(sent: str, lang: str, keep_tokens: bool = False) -> Union[str, List[str]]:
    if f"{lang}-tok-only" not in _LOADED_NLP:
        _LOADED_NLP[f"{lang}-tok-only"] = load_nlp(lang, tok_only=True)
    nlp = _LOADED_NLP[f"{lang}-tok-only"]
    # Since spaces are used to separate tokens in TERCom, if some are left inside tokens (e.g. "400 000")
    # they are converted to underscores. This is better than removing them, since it preserves the length
    # of the token and hence character offsets.
    tokens = [token.text.replace(" ", "_") for s in nlp(sent).sentences for token in s.tokens]
    if keep_tokens:
        return tokens
    return " ".join(tokens)


def fill_blanks(annotation: dict) -> dict:
    return {
        field: annotation.get(field, "") if t is str else annotation.get(field, -1)
        for field, t in STANZA_FIELDS.items()
    }


def merge_mwt(annotation: List[dict]) -> dict:
    if len(annotation) < 1:
        raise ValueError("Annotation must have at least one token")
    annotation = [fill_blanks(tok) for tok in annotation]
    # single-word token
    if len(annotation) == 1:
        return annotation[0]
    # multi-word token
    merged = {}
    token = annotation[0]
    words = annotation[1:]
    for field in STANZA_WORD_FIELDS:
        merged[field] = "+".join([str(word[field]) for word in words])
    for field in STANZA_TOKEN_FIELDS:
        merged[field] = token[field]
    return merged


def get_tokens_annotations(text: Optional[str], lang: str) -> Tuple[Optional[List[str]], Optional[List[dict]]]:
    if lang not in _LOADED_NLP:
        _LOADED_NLP[lang] = load_nlp(lang)
    nlp = _LOADED_NLP[lang]
    if nlp is None or text is None:
        return None, None
    doc = nlp(text)
    tokens = []
    annotations = []
    for s in doc.sentences:
        for tok in s.tokens:
            tokens.append(tok.text)
            annotations.append(merge_mwt(tok.to_dict()))
    return tokens, annotations


@CacheDecorator()
def texts2annotations(data: pd.DataFrame, unit_id_contains_lang: bool = True) -> pd.DataFrame:
    if "lang_id" not in data.columns and unit_id_contains_lang:
        data["lang_id"] = data.unit_id.str.split("-").map(lambda x: x[2])

    src_tokens, mt_tokens, tgt_tokens = [], [], []
    src_annotations, mt_annotations, tgt_annotations = [], [], []

    for _i, row in tqdm(data.iterrows(), desc="Adding Stanza annotations...", total=len(data)):
        src_tok, src_ann = get_tokens_annotations(row.src_text, "eng")
        mt_tok, mt_ann = get_tokens_annotations(row.mt_text, row.lang_id)
        tgt_tok, tgt_ann = get_tokens_annotations(row.tgt_text, row.lang_id)
        src_tokens.append(src_tok)
        src_annotations.append(src_ann)
        mt_tokens.append(mt_tok)
        mt_annotations.append(mt_ann)
        tgt_tokens.append(tgt_tok)
        tgt_annotations.append(tgt_ann)
    data["src_tokens"] = src_tokens
    data["src_annotations"] = src_annotations
    data["mt_tokens"] = mt_tokens
    data["mt_annotations"] = mt_annotations
    data["tgt_tokens"] = tgt_tokens
    data["tgt_annotations"] = tgt_annotations

    clear_nlp_cache()

    return data
