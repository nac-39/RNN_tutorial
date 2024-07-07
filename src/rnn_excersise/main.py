import argparse
import json
from logging import getLogger, config
import torch

from rnn_excersise.data import DataLoader
from rnn_excersise.train import Train


def get_config(path="logger_config.json"):
    with open(path, "r") as f:
        file = json.load(f)
        config.dictConfig(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", help="retrain model")
    parser.add_argument("--n_iters", help="number of iterations")
    parser.add_argument("--learning_rate", help="learning rate")
    parser.add_argument("--num_layers", help="number of hidden layers")
    parser.add_argument("--max_norm", help="max norm")
    args = parser.parse_args()

    get_config()
    logger = getLogger(__name__)
    logger.info("-" * 100)

    if torch.cuda.is_available():
        d_type = "cuda:0"
        logger.info("GPU is available.")
    else:
        d_type = "cpu"
        logger.info("GPU is not available.")

    device = torch.device(d_type)

    # データの読み込み
    loader = DataLoader("data", "anime-dataset-2023.csv", "English name")
    loader.load_data()
    data = loader.parse_data(device)

    # モデルの訓練
    learning_rate = float(args.learning_rate) if args.learning_rate else 0.005
    n_iters = int(args.n_iters) if args.n_iters else 500000
    max_norm = float(args.max_norm) if args.max_norm else 5.0
    num_layers = int(args.num_layers) if args.num_layers else 3

    logger.info(f"learning_rate: {learning_rate}")
    logger.info(f"n_iters: {n_iters}")
    logger.info(f"max_norm: {max_norm}")
    logger.info(f"num_layers: {num_layers}")

    trainer = Train(
        data,
        device,
        logger,
        n_hidden=128,
        learning_rate=learning_rate,
        num_layers=num_layers,
    )
    trained_model, all_losses = trainer.proceed(
        retrain=True, n_iters=n_iters, max_norm=max_norm
    )
