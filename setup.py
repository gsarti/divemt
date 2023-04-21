from setuptools import setup

setup(
    name="divemt",
    version="0.2.0",
    author="Gabriele Sarti",
    author_email="g.sarti@rug.nl",
    description="Post-editing effectiveness for typologically-diverse languages",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="GPLv3",
    url="https://github.com/gsarti/divemt",
    package_dir={"": "."},
    packages=["divemt"],
    install_requires=[
        "numpy<1.19.5",  # as simalign is not compatible with numpy >=1.20.0 (np.int is deprecated), 1.19.5 vulnerable
        "pandas",
        "sacrebleu",
        "Levenshtein",
        "stanza",
        "simalign",
        "strenum",
        "sentencepiece",
        "tqdm",
        "black",
        "flake8",
        "isort",
        "pytest",
        "ruff",
    ],
    python_requires=">=3.8.0",
)
