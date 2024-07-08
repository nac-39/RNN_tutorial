#!/bin/bash

if [ ! -f ./myanimelist-dataset.zip ] || [ ! -f ./data/myanimelist-dataset.zip ]; then
    kaggle datasets download -d dbdmobile/myanimelist-dataset
fi
SCRIPT_DIR=$(dirname "$(realpath "$0")")

rye run python "$SCRIPT_DIR/main.py" \
    --retrain \
    --n_iters 10000 \
    --learning_rate 0.0001 \
    --num_layers 3 \
    --max_norm 5.0

rye run python "$SCRIPT_DIR/main.py" \
    --retrain \
    --n_iters 100000 \
    --learning_rate 0.0001 \
    --num_layers 3 \
    --max_norm 5.0

rye run python "$SCRIPT_DIR/plot.py" 10000\
    --model "$SCRIPT_DIR/models/20240708121125" \
