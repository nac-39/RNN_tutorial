import argparse
import torch

from .data import DataLoader
from .train import Train

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
    loader = DataLoader("data", "", "English Name")
    loader.load_data()
    data = loader.parse_data(device)

    # モデルの訓練
    trainer = Train(data, device, learning_rate=args.retrain)
    trained_model, all_losses = trainer.proceed(retrain=True, n_iters=int(args.n_iters))
