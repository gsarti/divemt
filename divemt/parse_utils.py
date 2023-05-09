"""
Utilities for parsing and processing the data from .per to tabular format.
Includes utilities for computing different sentence-level metrics (BLEU, TER, ChrF, CER).
"""

import codecs
import ctypes
import logging
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd
from sacrebleu import sentence_bleu, sentence_chrf

from .cer import cer
from .tag_utils import clear_nlp_cache, texts2annotations, tokenize

from .qe_taggers import QETagger, WMT22QETagger, NameTBDTagger  # isort: skip  <- due to circular import with tag_utils

logger = logging.getLogger(__name__)

_KEYS_INDICATOR_MAP = {
    "k_letter": "indicator[@id='letter-keys']",
    "k_digit": "indicator[@id='digit-keys']",
    "k_white": "indicator[@id='white-keys']",
    "k_symbol": "indicator[@id='symbol-keys']",
    "k_nav": "indicator[@id='navigation-keys']",
    "k_erase": "indicator[@id='erase-keys']",
    "k_copy": "indicator[@id='copy-keys']",
    "k_cut": "indicator[@id='cut-keys']",
    "k_paste": "indicator[@id='paste-keys']",
    "k_do": "indicator[@id='do-keys']",
}

_METRICS_DF_FIELDS = [
    "unit_id",
    "flores_id",
    "item_id",
    "subject_id",
    "task_type",
    "translation_type",
    "src_len_chr",
    "mt_len_chr",
    "tgt_len_chr",
    "src_len_wrd",
    "mt_len_wrd",
    "tgt_len_wrd",
    "edit_time",
    "k_total",
    "k_letter",
    "k_digit",
    "k_white",
    "k_symbol",
    "k_nav",
    "k_erase",
    "k_copy",
    "k_cut",
    "k_paste",
    "k_do",
    "n_pause_geq_300",
    "len_pause_geq_300",
    "n_pause_geq_1000",
    "len_pause_geq_1000",
    "event_time",
    "num_annotations",
    "last_modification_time",
]

_EDITS_DF_MAP = {
    "Ins": "n_insert",
    "Del": "n_delete",
    "Sub": "n_substitute",
    "Shft": "n_shift",
    "WdSh": "tot_shifted_words",
    "NumEr": "tot_edits",
    "NumWd": "n_words",
    "TER": "hter",
}

_EDITS_DF_TYPES = {v: int for v in _EDITS_DF_MAP.values() if v != "hter"}


def time2seconds(_input: str) -> float:
    """Convert a time expression from a per file to seconds.
    e.g. 1m15s,719 -> 75.719
    """
    hours = 0
    minutes = 0
    seconds = 0
    milliseconds = 0
    if "h" in _input:
        hours, _input = tuple(_input.split("h"))
    if "m" in _input:
        minutes, _input = tuple(_input.split("m"))
    if "s" in _input:
        seconds, _input = tuple(_input.split("s"))
    if "," in _input:
        milliseconds = _input.split(",")[1]
    hours, minutes, seconds, milliseconds = map(int, (hours, minutes, seconds, milliseconds))
    return round(hours * 3600 + minutes * 60 + seconds + milliseconds / 1000, 3)


def per2metrics(filename: str) -> Optional[pd.DataFrame]:
    """Convert an XML PET result file to a Pandas DataFrame."""
    tree = ET.ElementTree(file=filename)
    if tree.getroot().get("status") != "FINISHED":
        logger.warning(f"KAPUT: {filename}", tree.getroot().attrib)
        return None
    job_id = tree.getroot().get("id")
    dict_metrics = defaultdict(list)
    for curr_unit in tree.iterfind("unit"):
        curr_unit_id = curr_unit.get("id")
        flores_id = curr_unit.get("orig_id")
        translation_type = curr_unit.get("type")
        dict_metrics["unit_id"] += [f"{job_id}-{curr_unit_id}"]
        dict_metrics["flores_id"] += [flores_id]
        dict_metrics["item_id"] += [job_id.rsplit("-", 1)[0] + curr_unit_id]
        dict_metrics["task_type"] += [job_id.rsplit("-", 1)[1]]
        dict_metrics["translation_type"] += [translation_type]
        source_text = curr_unit.findtext("S")
        dict_metrics["src_len_chr"] += [len(source_text)]
        dict_metrics["src_len_wrd"] += [len(source_text.split())]
        if translation_type == "pe":
            mt_text = curr_unit.findtext("MT")
            dict_metrics["mt_len_chr"] += [len(mt_text)]
            dict_metrics["mt_len_wrd"] += [len(mt_text.split())]
        else:
            dict_metrics["mt_len_chr"] += [None]
            dict_metrics["mt_len_wrd"] += [None]
        annotations = list(curr_unit.iterfind("annotations/annotation"))
        final_annotation = annotations[-1]
        target = final_annotation.find("PE" if translation_type == "pe" else "HT")
        dict_metrics["subject_id"] += [target.get("producer").split(".")[0]]
        dict_metrics["tgt_len_chr"] += [len(target.text)]
        dict_metrics["tgt_len_wrd"] += [len(target.text.split())]
        dict_metrics["edit_time"] += [
            round(
                sum([time2seconds(ann.findtext("indicator[@id='editing']")) for ann in annotations]),
                3,
            )
        ]
        for field, xpath in _KEYS_INDICATOR_MAP.items():
            dict_metrics[field] += [sum([int(ann.findtext(xpath)) for ann in annotations])]
        dict_metrics["k_total"] += [sum([dict_metrics[k][-1] for k in _KEYS_INDICATOR_MAP.keys()])]
        event_times = [
            [int(event.get("t")) for event in list(annotation.find("events"))] for annotation in annotations
        ]

        # We only consider pauses between elements keystroke and command
        filtered_event_times = [
            [int(event.get("t")) for event in list(annotation.find("events")) if event.tag in ["keystroke", "command"]]
            for annotation in annotations
        ]
        dict_metrics["event_time"] += [
            sum([ann_events[-1] if len(ann_events) > 0 else 0 for ann_events in event_times])
        ]
        pauses = [
            [next_t - curr_t for curr_t, next_t in zip([0] + ann_events, ann_events)]
            for ann_events in filtered_event_times
        ]
        dict_metrics["n_pause_geq_300"] += [
            sum([sum([1 if pause >= 300 else 0 for pause in ann_pauses]) for ann_pauses in pauses])
        ]
        dict_metrics["len_pause_geq_300"] += [
            sum([sum([pause for pause in ann_pauses if pause >= 300]) for ann_pauses in pauses])
        ]
        dict_metrics["n_pause_geq_1000"] += [
            sum([sum([1 if pause >= 1000 else 0 for pause in ann_pauses]) for ann_pauses in pauses])
        ]
        dict_metrics["len_pause_geq_1000"] += [
            sum([sum([pause for pause in ann_pauses if pause >= 1000]) for ann_pauses in pauses])
        ]
        dict_metrics["num_annotations"] += [len(annotations)]
        dict_metrics["last_modification_time"] += [int(os.path.getmtime(filename))]

    # Debug edit_time vs event_time
    edit_times_ms = [float(edit_time) * 1000 for edit_time in dict_metrics["edit_time"]]
    for idx, (edit_time, event_time) in enumerate(zip(edit_times_ms, dict_metrics["event_time"])):
        if edit_time < float(event_time) * 0.995 or edit_time > float(event_time) * 1.005:
            rel_diff = (float(event_time) - edit_time) / edit_time
            logger.warning(
                f"{dict_metrics['unit_id'][idx]}: edit_time {edit_time} vs event_time "
                f"{float(event_time)} ({rel_diff:.2%} diff, {dict_metrics['num_annotations'][idx]} annotations)"
            )
    metrics_df = pd.DataFrame(dict_metrics)
    return metrics_df[_METRICS_DF_FIELDS]


def per2texts(filename: str) -> Optional[pd.DataFrame]:
    """Convert an XML PET result file to a Pandas DataFrame."""
    tree = ET.ElementTree(file=filename)
    if tree.getroot().get("status") != "FINISHED":
        logger.warning(f"KAPUT: {filename}", tree.getroot().attrib)
        return None
    job_id = tree.getroot().get("id")
    dict_texts = defaultdict(list)
    for curr_unit in tree.iterfind("unit"):
        curr_unit_id = curr_unit.get("id")
        trans_type = curr_unit.get("type")
        dict_texts["unit_id"] += [f"{job_id}-{curr_unit_id}"]
        dict_texts["src_text"] += [curr_unit.findtext("S")]
        dict_texts["mt_text"] += [curr_unit.findtext("MT")] if trans_type == "pe" else [None]
        final_annotation = list(curr_unit.iterfind("annotations/annotation"))[-1]
        dict_texts["tgt_text"] += [final_annotation.findtext("PE" if trans_type == "pe" else "HT")]
    texts_df = pd.DataFrame(dict_texts)
    return texts_df


def texts2cer(
    ref_sentences: List[str],
    hyp_sentences: List[str],
    libed_path: str = "scripts/libED.so",
) -> List[float]:
    # Initialise the connection to C++
    ed_wrapper = ctypes.CDLL(libed_path)
    ed_wrapper.wrapper.restype = ctypes.c_float
    scores = []
    # Split the hypothesis and reference sentences into word lists
    for _, (hyp, ref) in enumerate(zip(hyp_sentences, ref_sentences), start=1):
        _ref, _hyp = ref.split(), hyp.split()
        score = cer(_hyp, _ref, ed_wrapper)
        scores.append(score)
    return scores


def texts2edits(
    data: Optional[pd.DataFrame] = None,
    ref_name: str = "mt_text",
    hyp_name: str = "tgt_text",
    id_name: str = "unit_id",
    tercom_path: str = "scripts/tercom.7.25.jar",
    unit_id_contains_lang: bool = True,
) -> pd.DataFrame:
    tmp_path = "tmp"
    prefix = "tmp_tercom_out"
    os.makedirs(tmp_path, exist_ok=True)
    if data is not None:
        if unit_id_contains_lang:
            lang_ids = list(data.unit_id.str.split("-").map(lambda x: x[2]))
        refs, hyps, ids = [], [], []
        for i, r in data.iterrows():
            if r[ref_name] is not None and r[ref_name].strip() != "-":
                assert "pe" in r[id_name]
                refs += [tokenize(r[ref_name], lang_ids[i])] if lang_ids else [r[ref_name]]
                hyps += [tokenize(r[hyp_name], lang_ids[i])] if lang_ids else [r[hyp_name]]
                ids += [r[id_name]]
        # Prepare files for tercom
        ref_fname = os.path.join(tmp_path, f"{prefix}_ref.txt")
        hyp_fname = os.path.join(tmp_path, f"{prefix}_hyp.txt")
        with codecs.open(ref_fname, "w", encoding="utf-8") as rf:
            with codecs.open(hyp_fname, "w", encoding="utf-8") as hf:
                for ref, hyp, idx in zip(refs, hyps, ids):
                    _ref = ref.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', '\\"')
                    _hyp = hyp.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', '\\"')
                    rf.write(f"{_ref}\t({idx})\n")
                    hf.write(f"{_hyp}\t({idx})\n")
    else:
        ref_fname, hyp_fname = ref_name, hyp_name
    out_rootname = os.path.join(tmp_path, prefix)
    try:
        tercom_params = [
            "java",
            "-jar",
            tercom_path,
            "-r",
            ref_fname,
            "-h",
            hyp_fname,
            "-n",
            out_rootname,
        ]
        _ = subprocess.run(tercom_params, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"Error while running tercom: {e.stderr}.\nPlease make sure you have java installed and that the .jar "
            f"file is found at {tercom_path}"
        )
    # Parse tercom output
    ter_metrics = pd.read_table(f"{out_rootname}.sum", skiprows=3, sep="|", header=0, skipinitialspace=True).iloc[
        1:-2, :
    ]
    ter_metrics.columns = [x.strip() for x in ter_metrics.columns]
    ter_metrics["Sent Id"] = ter_metrics["Sent Id"].apply(lambda x: x.split(":")[0])
    with open(f"{out_rootname}.pra_more") as f:
        aligned_edits = "\n".join([x.strip() for x in f.readlines()])
    shutil.rmtree(tmp_path)
    p = re.compile("REF:.*\nHYP:.*\nEVAL:.*")
    ter_metrics["aligned_edit"] = [x.replace("\n", "\\n") for x in p.findall(aligned_edits)]
    ter_metrics = ter_metrics.rename(columns={**{"Sent Id": id_name}, **_EDITS_DF_MAP}).astype(_EDITS_DF_TYPES)
    # Intentionally swapped to match the expected input of CER
    try:
        ter_metrics["cer"] = texts2cer(hyps, refs)
    except Exception as e:
        # Not working at macos
        logger.warning(f"Error while computing CER: {e}")
    clear_nlp_cache()
    return ter_metrics


def texts2scores(
    data: Optional[pd.DataFrame] = None,
    ref_name: Union[str, Sequence[str]] = "tgt_text",
    hyp_name: Union[str, Sequence[str]] = "mt_text",
    id_name: Optional[str] = "unit_id",
) -> pd.DataFrame:
    """Compute BLEU, and chrF scores for reference-hypothesis pairs."""
    if data is not None:
        hyps = data[hyp_name]
        refs = data[ref_name]
    metric_vals = defaultdict(list)
    if id_name is not None:
        metric_vals[id_name] = data[id_name]
    for metric, score_func in {"bleu": sentence_bleu, "chrf": sentence_chrf}.items():
        for h, r in zip(hyps, refs):
            if h is not None and r is not None:
                metric_vals[metric] += [round(score_func(h, [r]).score, 2)]
            else:
                metric_vals[metric] += [None]
    return pd.DataFrame(metric_vals)


def metrics2extra(
    metrics_df: pd.DataFrame,
    unit_id_contains_lang: bool = True,
    unit_id_contains_doc: bool = True,
):
    """Add extra metrics that can be derived from other metrics."""
    if unit_id_contains_lang:
        metrics_df["lang_id"] = metrics_df.unit_id.str.split("-").map(lambda x: x[2])
    if unit_id_contains_doc:
        metrics_df["doc_id"] = metrics_df.unit_id.str.split("-").map(lambda x: x[3])
    metrics_df["time_s"] = metrics_df.event_time / 1000
    metrics_df["time_m"] = metrics_df.time_s / 60
    metrics_df["time_h"] = metrics_df.time_s / 3600
    metrics_df["time_per_char"] = metrics_df.time_s / metrics_df.src_len_chr
    metrics_df["time_per_word"] = metrics_df.time_s / metrics_df.src_len_wrd
    metrics_df["key_per_char"] = metrics_df.k_total / metrics_df.src_len_chr
    metrics_df["words_per_hour"] = metrics_df.src_len_wrd / metrics_df.time_h
    metrics_df["words_per_minute"] = metrics_df.src_len_wrd / metrics_df.time_m
    return metrics_df


def texts2qe(
    data: pd.DataFrame,
    tagger: QETagger,
) -> pd.DataFrame:
    """Add quality tags to a dataframe."""
    pe_texts = data.copy()[data.mt_text.notnull()]
    src_tags, mt_tags = tagger.generate_tags(
        pe_texts["src_text"].tolist(),
        pe_texts["mt_text"].tolist(),
        pe_texts["tgt_text"].tolist(),
        "eng",
        pe_texts.unit_id.str.split("-").map(lambda x: x[2]).tolist(),
    )
    pe_texts[f"src_{tagger.ID}"] = src_tags
    pe_texts[f"mt_{tagger.ID}"] = mt_tags
    pe_texts = pe_texts[["unit_id", f"src_{tagger.ID}", f"mt_{tagger.ID}"]]
    data = data.join(pe_texts.set_index("unit_id"), on="unit_id")
    return data


def parse_from_folder(
    path: str,
    ref_name: Union[str, Sequence[str]] = "tgt_text",
    hyp_name: Union[str, Sequence[str]] = "mt_text",
    id_name: Optional[str] = "unit_id",
    time_ordered: bool = True,
    output_texts: bool = False,
    add_edit_information: bool = False,
    add_eval_information: bool = False,
    add_extra_information: bool = False,
    add_annotations_information: bool = False,
    add_wmt22_quality_tags: bool = False,
    add_name_tbd_quality_tags: bool = False,
    rounding: Optional[int] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Parse all .per XML files in a folder and return a single dataframe containing all units."""
    metrics_list_dfs = [per2metrics(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".per")]
    metrics_df = pd.concat([df for df in metrics_list_dfs if df is not None], ignore_index=True)
    if (
        output_texts
        or add_edit_information
        or add_eval_information
        or add_extra_information
        or add_annotations_information
    ):
        texts_list_dfs = [per2texts(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".per")]
        texts_df = pd.concat([df for df in texts_list_dfs if df is not None], ignore_index=True)
        if add_edit_information:
            # Needed for HTER (using human PE as reference)
            edits_df = texts2edits(texts_df, hyp_name, ref_name, id_name)
            metrics_df = pd.merge(metrics_df, edits_df, how="left", on=id_name)
            texts_df["aligned_edit"] = list(metrics_df["aligned_edit"])
            metrics_df.drop(columns=["aligned_edit", "n_words"], inplace=True)
        if add_eval_information:
            scores_df = texts2scores(texts_df, ref_name, hyp_name, id_name)
            if id_name in scores_df.columns:
                metrics_df = pd.merge(metrics_df, scores_df, how="left", on=id_name)
            else:
                metrics_df = pd.concat([metrics_df, scores_df], axis=1, ignore_index=True)
        if add_extra_information:
            metrics_df = metrics2extra(metrics_df)
        if add_annotations_information:
            texts_df = texts2annotations(texts_df)  # TODO: make cache optional
        if add_wmt22_quality_tags:
            tagger = WMT22QETagger()
            texts_df = texts2qe(texts_df, tagger)
        if add_name_tbd_quality_tags:
            tagger = NameTBDTagger()  # TODO: make cache optional
            texts_df = texts2qe(texts_df, tagger)

    if time_ordered:
        if output_texts:
            texts_df["time"] = metrics_df["last_modification_time"]
            texts_df = texts_df.sort_values(by=["time", "unit_id"]).drop(columns=["time"])
        metrics_df = metrics_df.sort_values(by=["last_modification_time", "unit_id"])
        metrics_df["per_subject_visit_order"] = list(range(1, len(metrics_df) + 1))

    if rounding is not None:
        for col in metrics_df.select_dtypes(include=["float"]).columns:
            metrics_df[col] = metrics_df[col].round(rounding)

    if output_texts:
        return metrics_df, texts_df

    return metrics_df
