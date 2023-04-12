import argparse
import logging
import os
from itertools import product

import pandas as pd

from divemt import parse_from_folder

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%d-%m-%y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def preprocess(args: argparse.Namespace):
    """
    Preprocess raw PER data
    """
    if args.languages is None:
        args.languages = [f for f in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, f))]
    lang_source_paths = {lang: os.path.join(args.data_dir, lang) for lang in args.languages}
    lang_output_paths = {lang: os.path.join(args.output_dir, lang) for lang in args.languages}
    if args.output_single:
        for path in lang_output_paths.values():
            os.makedirs(path, exist_ok=True)
    if args.tasks is None:
        tasks = {
            lang: list(set([f.split("_")[1] for f in os.listdir(lang_source_paths[lang])])) for lang in args.languages
        }
    else:
        tasks = {lang: args.tasks for lang in args.languages}
    results_dict = {lang: {task: [] for task in tasks[lang]} for lang in args.languages}
    texts_dict = {lang: {task: [] for task in tasks[lang]} for lang in args.languages}
    logger.info(args)
    for lang in args.languages:
        logger.info(f"\nLanguage: {lang}\n========")
        subjects = (
            list(sorted(set([f.split("_")[0] for f in os.listdir(lang_source_paths[lang])])))
            if args.subjects is None
            else args.subjects
        )
        for subject, task in product(subjects, tasks[lang]):
            subj_task_path = os.path.join(lang_source_paths[lang], subject + "_" + task)
            if os.path.exists(subj_task_path):
                logger.info(f"Processing {subj_task_path}")
                df_metrics = parse_from_folder(
                    subj_task_path,
                    output_texts=args.output_texts,
                    add_edit_information=args.add_edits,
                    add_eval_information=args.add_evals,
                    add_extra_information=args.add_extra,
                    add_annotations_information=args.add_annotations,
                    add_wmt22_quality_tags=args.add_wmt22_quality_tags,
                    rounding=args.rounding,
                )
                if args.output_texts:
                    df_metrics, df_texts = df_metrics
                if df_metrics["subject_id"].iloc[0] != subject:
                    logger.warning(
                        f"Subject ID mismatch: {subject} != {df_metrics['subject_id'].iloc[0]}. Replacing the subject."
                    )
                    df_metrics["subject_id"] = subject
                if args.output_single:
                    df_metrics.to_csv(
                        os.path.join(lang_output_paths[lang], f"{subject}_{task}.tsv"),
                        sep="\t",
                        index=False,
                    )
                    if args.output_texts:
                        df_texts.to_csv(
                            os.path.join(args.output_dir, lang, f"{subject}_{task}_texts.tsv"),
                            sep="\t",
                            index=False,
                        )
                df_metrics["subject_id"] = lang + "_" + df_metrics["subject_id"]
                if f"-{lang}-" in df_metrics["item_id"].iloc[0]:
                    df_metrics["item_id"] = df_metrics["item_id"].str.replace(f"-{lang}-", "-")
                results_dict[lang][task].append(df_metrics)
                texts_dict[lang][task].append(df_texts)
        if args.output_merged_subjects or args.output_merged_languages:
            merged_dir = os.path.join(args.output_dir, "merged")
            os.makedirs(merged_dir, exist_ok=True)
        if args.output_merged_subjects:
            for task, dfs in results_dict[lang].items():
                df_metrics = pd.concat(dfs)
                df_metrics.to_csv(
                    os.path.join(merged_dir, f"{lang}_{'_'.join(subjects)}_{task}.tsv"),
                    sep="\t",
                    index=False,
                )
            for task, dfs in texts_dict[lang].items():
                df_texts = pd.concat(dfs)
                df_texts.to_csv(
                    os.path.join(merged_dir, f"{lang}_{'_'.join(subjects)}_{task}_texts.tsv"),
                    sep="\t",
                    index=False,
                )
    if args.output_merged_languages:
        all_tasks = set([task for lang in args.languages for task in tasks[lang]])
        for task in all_tasks:
            df_metrics = pd.concat(
                [df_metrics for lang in args.languages for df_metrics in results_dict[lang].get(task, [])]
            )
            df_metrics.to_csv(os.path.join(merged_dir, f"full_{task}.tsv"), sep="\t", index=False)
            df_texts = pd.concat([df_texts for lang in args.languages for df_texts in texts_dict[lang].get(task, [])])
            df_texts.to_csv(
                os.path.join(merged_dir, f"full_{task}_texts.tsv"),
                sep="\t",
                index=False,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        default="data/raw",
        help="Folder containing language subfolders",
    )
    parser.add_argument(
        "--languages",
        "-l",
        nargs="+",
        type=str,
        default=None,
        help="Languages to preprocess",
    )
    parser.add_argument(
        "--tasks",
        "-t",
        nargs="+",
        type=str,
        default=None,
        help="Settings to preprocess",
    )
    parser.add_argument(
        "--subjects",
        "-s",
        nargs="+",
        type=str,
        default=None,
        help="Subjects to preprocess",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data/processed",
        help="Folder to save preprocessed data",
    )
    parser.add_argument(
        "--output_texts",
        action="store_true",
        help="Whether a tsv containing texts should be saved alongside metrics",
    )
    parser.add_argument(
        "--add_edits",
        action="store_true",
        help="Whether edit information should be added to the metrics dataframe",
    )
    parser.add_argument(
        "--add_evals",
        action="store_true",
        help="Whether evaluation metrics should be added to the metrics dataframe",
    )
    parser.add_argument(
        "--add_extra",
        action="store_true",
        help="Whether to add extra information computed from existing fields",
    )
    parser.add_argument(
        "--add_annotations",
        action="store_true",
        help="Whether to add annotation information computed from Stanza",
    )
    parser.add_argument(
        "--add_wmt22_quality_tags",
        action="store_true",
        help="Whether to add WMT22 quality tags to the text dataframe",
    )
    parser.add_argument(
        "--output_single",
        action="store_true",
        help="Whether individual dataframes should be saved for each task-subject pair",
    )
    parser.add_argument(
        "--output_merged_subjects",
        action="store_true",
        help="Whether a merged dataframe across subjects should be saved",
    )
    parser.add_argument(
        "--output_merged_languages",
        action="store_true",
        help="Whether a merged dataframe across subjects and languages should be saved",
    )
    parser.add_argument(
        "--rounding",
        type=int,
        default=4,
        help="Decimals to round for floating point scores. Default: 4",
    )
    args = parser.parse_args()
    preprocess(args)
