from setuptools import setup

setup(
    name="divemt",
    version="0.1.0",
    author="Gabriele Sarti",
    author_email="g.sarti@rug.nl",
    description="Post-editing effectiveness for typologically-diverse languages",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="GPLv3",
    url="https://github.com/gsarti/divemt",
    package_dir={"": "."},
    packages=["divemt"],
    install_requires=[
        "pandas",
        "sacrebleu",
        "Levenshtein",
        "stanza",
        "black",
        "flake8",
        "isort",
    ],
    python_requires=">=3.6.0"
)