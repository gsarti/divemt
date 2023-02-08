"""Utilities for tagging and tokenization."""

from typing import Optional

import stanza
import logging
import pandas as pd
from tqdm import tqdm

_STANZA_NLP_MAP = {
    "eng": {"lang":'en', "processors":'tokenize,pos,depparse,ner,lemma'},
    "ara": {"lang":'ar', "processors":'tokenize,pos,depparse,ner,lemma,mwt'},
    "nld": {"lang":'nl', "processors":'tokenize,pos,depparse,ner,lemma'},
    "ita": {"lang":'it', "processors":'tokenize,pos,depparse,ner,lemma,mwt'},
    "tur": {"lang":'tr', "processors":'tokenize,pos,depparse,ner,lemma,mwt'},
    "ukr": {"lang":'uk', "processors":'tokenize,pos,depparse,ner,lemma,mwt'},
    "vie": {"lang":'vi', "processors":'tokenize,pos,depparse,ner,lemma'},
}

stanza.logger.setLevel(logging.WARNING)

_LOADED_NLP = {}

STANZA_FIELDS = {
    "id": int,
    "text": str,
    "lemma": str,
    "upos": str,
    "xpos": str,
    "feats": str,
    "head": int,
    "deprel": str,
    "start_char": int,
    "end_char": int,
    "ner": str,
}

def load_nlp(lang: str, tok_only: bool = False):
    if lang not in _STANZA_NLP_MAP:
        raise ValueError(f"Language {lang} not supported")
    if tok_only:
        return stanza.Pipeline(lang=_STANZA_NLP_MAP[lang]["lang"], processors='tokenize')
    return stanza.Pipeline(lang=_STANZA_NLP_MAP[lang]["lang"], processors=_STANZA_NLP_MAP[lang]["processors"])

def clear_nlp_cache():
    _LOADED_NLP.clear()

def tokenize(sent: str, lang: str):
    if f"{lang}-tok-only" not in _LOADED_NLP:
        _LOADED_NLP[f"{lang}-tok-only"] = load_nlp(lang, tok_only=True)
    nlp = _LOADED_NLP[f"{lang}-tok-only"]
    return " ".join([token.text for s in nlp(sent).sentences for token in s.tokens])

def format_annotation(annotation: dict) -> str:
    return {
        field: annotation.get(field, '') 
        if t is str else annotation.get(field, -1) 
        for field, t in STANZA_FIELDS.items()
    }

def get_tokens_annotations(text: Optional[str], lang: str):
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
            annotations.append([format_annotation(x) for x in tok.to_dict()])
    return tokens, annotations

def texts2annotations(
    data: pd.DataFrame,
    unit_id_contains_lang: bool = True
) -> pd.DataFrame:
    if "lang_id" not in data.columns and unit_id_contains_lang:
        data["lang_id"] = data.unit_id.str.split("-").map(lambda x: x[2])
    src_tokens = []
    src_annotations = []
    mt_tokens = []
    mt_annotations = []
    tgt_tokens = []
    tgt_annotations = []
    for i, row in tqdm(data.iterrows(), desc="Adding Stanza annotations...", total=len(data)):
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