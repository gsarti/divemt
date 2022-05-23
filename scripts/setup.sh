#!/bin/bash

# Unzip the zipped data folders
for folder in data/raw/*.zip; do
    unzip $folder -d data/raw/
done 