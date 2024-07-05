import argparse
import torch

from rnn_excersise.data import DataLoader
from rnn_excersise.train import Train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", help="retrain model")
    parser.add_argument("--n_iters", help="number of iterations")
    parser.add_argument("--learning_rate", help="learning rate")
    args = parser.parse_args()

    if torch.cuda.is_available():
        d_type = "cuda:0"
    else:
        d_type = "cpu"

    device = torch.device(d_type)

    # データの読み込み
    loader = DataLoader("data", "anime-dataset-2023.csv", "English name")
    loader.load_data()
    data = loader.parse_data(device)

    # モデルの訓練
    learning_rate = float(args.learning_rate) if args.learning_rate else 0.005
    n_iters = int(args.n_iters) if args.n_iters else 500000
    trainer = Train(data, device, n_hidden=128, learning_rate=learning_rate)
    trained_model, all_losses = trainer.proceed(retrain=True, n_iters=n_iters)
