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
        args.languages = [f for f in os.listdir(args.data_dir)]
    for lang in args.languages:
        logger.info(f"\nLanguage: {lang}\n========")
        lang_dir = os.path.join(args.data_dir, lang)
        os.makedirs(os.path.join(args.output_dir, lang), exist_ok=True)
        subjects = (
            list(sorted(set([f.split("_")[0] for f in os.listdir(lang_dir)])))
            if args.subjects is None
            else args.subjects
        )
        tasks = list(set([f.split("_")[1] for f in os.listdir(lang_dir)])) if args.tasks is None else args.tasks
        task_dict = {task: [] for task in tasks}
        for subject, task in product(subjects, tasks):
            subj_task_path = os.path.join(lang_dir, subject + "_" + task)
            if os.path.exists(subj_task_path):
                logger.info(f"Processing {subj_task_path}")
                df_metrics = parse_from_folder(
                    subj_task_path,
                    output_texts=args.output_texts,
                    add_edit_information=args.add_edits,
                    add_eval_information=args.add_evals,
                )
                if args.output_texts:
                    df_metrics, df_texts = df_metrics
                if df_metrics["subject_id"].iloc[0] != subject:
                    logger.warning(
                        f"Subject ID mismatch: {subject} != {df_metrics['subject_id'].iloc[0]}. Replacing the subject."
                    )
                    df_metrics["subject_id"] = subject
                df_metrics.to_csv(os.path.join(args.output_dir, lang, f"{subject}_{task}.tsv"), sep="\t", index=False)
                task_dict[task].append(df_metrics)
                if args.output_texts:
                    df_texts.to_csv(
                        os.path.join(args.output_dir, lang, f"{subject}_{task}_texts.tsv"), sep="\t", index=False
                    )
        if args.output_merged:
            merged_dir = os.path.join(args.output_dir, "merged")
            os.makedirs(merged_dir, exist_ok=True)
            for task in task_dict:
                df_metrics = pd.concat(task_dict[task])
                df_metrics.to_csv(
                    os.path.join(merged_dir, f"{lang}_{'_'.join(subjects)}_{task}.tsv"), sep="\t", index=False
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="data/raw", help="Folder containing language subfolders")
    parser.add_argument("--languages", "-l", nargs="+", type=str, default=None, help="Languages to preprocess")
    parser.add_argument("--tasks", "-t", nargs="+", type=str, default=None, help="Settings to preprocess")
    parser.add_argument("--subjects", "-s", nargs="+", type=str, default=None, help="Subjects to preprocess")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="data/processed", help="Folder to save preprocessed data"
    )
    parser.add_argument(
        "--output_texts", action="store_true", help="Whether a tsv containing texts should be saved alongside metrics"
    )
    parser.add_argument(
        "--add_edits", action="store_true", help="Whether edit information should be added to the metrics dataframe"
    )
    parser.add_argument(
        "--add_evals", action="store_true", help="Whether evaluation metrics should be added to the metrics dataframe"
    )
    parser.add_argument(
        "--output_merged", action="store_true", help="Whether a merged dataframe across subjects should be saved"
    )
    args = parser.parse_args()
    preprocess(args)
