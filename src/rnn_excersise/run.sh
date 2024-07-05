#!/bin/bash

if [ ! -f ./myanimelist-dataset.zip ] || [ ! -f ./data/myanimelist-dataset.zip ]; then
    kaggle datasets download -d dbdmobile/myanimelist-dataset
fi

rye run python ./main.py
