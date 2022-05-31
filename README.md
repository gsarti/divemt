# DivEMT: Neural Machine Translation Post-Editing Effort Across Typologically Diverse Languages

[Gabriele Sarti](https://gsarti.com) • [Arianna Bisazza](https://www.cs.rug.nl/~bisazza/) • [Ana Guerberof Arenas](https://scholar.google.com/citations?user=i6bqaTsAAAAJ) • [Antonio Toral](https://antoniotor.al/)

![DivEMT Annotation Pipeline](img/divemt.png)

> **Abstract:** We introduce DivEMT, the first publicly available post-editing study of Neural Machine Translation (NMT) over a typologically diverse set of target languages. Using a strictly controlled setup, 18 professional translators were instructed to translate or post-edit the same set of English documents into Arabic, Dutch, Italian, Turkish, Ukrainian, and Vietnamese. During the process, their edits, keystrokes, editing times, pauses, and perceived effort were logged, enabling an in-depth, cross-lingual evaluation of NMT quality and its post-editing process.
Using this new dataset, we assess the impact on translation productivity of two state-of-the-art NMT systems, namely: Google Translate and the open-source multilingual model mBART50. We find that, while post-editing is consistently faster than translation from scratch, the magnitude of its contribution varies largely across systems and languages, ranging from doubled productivity in Dutch and Italian to marginal gains by in Arabic, Turkish and Ukrainian, for some of the inspected modalities. Moreover, the observed cross-language variability appears to reflect source-target relatedness and type of target morphology, and it remains hard to predict based even by state-of-the-art automatic MT quality metrics. We publicly release the complete dataset including all collected behavioural data, to foster new research on the ability of state-of-the-art NMT systems to generate text in typologically diverse languages.

This repository contains data, scripts and notebooks associated to the paper ["DivEMT: Neural Machine Translation Post-Editing Effort Across Typologically Diverse Languages"](https://arxiv.org/abs/2205.12215). If you use any of the following contents for your work, we kindly ask you to cite our paper:

```bibtex
@article{sarti-etal-2022-divemt,
    title={{DivEMT}: Neural Machine Translation Post-Editing Effort Across Typologically Diverse Languages},
    author={Sarti, Gabriele and Bisazza, Arianna and Guerberof Arenas, Ana and Toral, Antonio},
    journal={ArXiv preprint 2205.12215},
    url={https://arxiv.org/abs/2205.12215},
    year={2022},
    month={may}
}
```


## DivEMT Explorer :mag:

The [DivEMT Explorer](https://huggingface.co/spaces/GroNLP/divemt_explorer) is a Streamlit demo hosted on [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/GroNLP/divemt_explorer) allowing for a seamless exploration of the different examples that compose the DivEMT dataset, across modalities and languages. Have a look!

## The DivEMT Dataset

The processed DivEMT dataset is accessible as a [🤗 Dataset](https://huggingface.co/datasets/GroNLP/divemt) via the GroNLP organization, or alternatively as a two TSV files (one with sentences and one with scores and metrics) in the [data](data/) folder. The raw `.per` files produced by PET are also released in the folder [data/raw](data/raw) to foster more fine-grained exploration of the translation process.

### Reproducing the Preprocessing

The procedure requires Python > 3.8 and a Java installation to run Tercom.

- Clone this repository and install it as a Python package:

```bash
git clone https://github.com/gsarti/divemt
cd divemt
pip install -e .
```

- Run the setup script to unzip all raw data folders.

- Run the preprocessing script to produce the TSV files containing sentences, scores and metrics.

```bash
# Run the preprocessing script
# --output_texts: Output TSV files with sentences (only scores otherwise)
# --add_edits: Add HTER and edit types breakdown to scores
# --add_evals: Add Bleu and ChrF to scores
# --add_extra: Add extra derived metrics to scores
# --output_single: Produces individual TSVs for every language-translator pair in the respective language folders
# --output_merged_subjects: Produces TSVs grouping all translators for every given language
# --output_merged_languages: Produces the final TSV with all languages and translators
python scripts/preprocess.py \
--output_texts \
--add_edits \
--add_evals \
--add_extra \
--output_single \
--output_merged_subjects \
--output_merged_languages
```

- The final data are produced as follows:

```python
import pandas as pd

main = pd.read_csv('data/processed/merged/full_main.tsv', sep="\t")
main_texts = pd.read_csv('data/processed/merged/full_main_texts.tsv', sep="\t")
warmup = pd.read_csv('data/processed/merged/full_warmup.tsv', sep="\t")
warmup_texts = pd.read_csv('data/processed/merged/full_warmup_texts.tsv', sep="\t")

df_main = pd.concat([main, main_texts.iloc[:, 1:]], axis=1)
# Avoid deduplication, rows are already matching
df_warmup = pd.concat([warmup, warmup_texts.iloc[:, 1:]], axis=1)

df_main.to_csv("data/main.tsv", sep="\t", index=False)
df_warmup.to_csv("data/warmup.tsv", sep="\t", index=False)
```

### Reproducing the Analysis and Plots

Follow along the [analysis](notebooks/analysis.ipynb) notebook after running the preprocessing script to produce plots and tables.

## Next steps

- [ ] Exhaustive data card for the DivEMT dataset

- [ ] Comments in source code for the DivEMT preprocessing methods

- [ ] Modeling notebook


## Dataset Curators

For any problem or question regarding DivEMT, please raise an issue in this repository.
