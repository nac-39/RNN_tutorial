from argparse import ArgumentParser
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from rnn_excersise.data import DataLoader
from rnn_excersise.predict import load_model, evaluate


def plot_all_losses(all_losses, save_path=None):
    plt.figure()
    plt.plot(all_losses)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_confusion_matrix(confusion, genres_list, save_path=None):
    plt.figure()
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    ax.set_xticklabels([""] + genres_list, rotation=90)
    ax.set_yticklabels([""] + genres_list)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("n_confusion", type=str, help="")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model directory"
    )
    args = parser.parse_args()

    loader = DataLoader("data", "anime-dataset-2023.csv", "English name")
    loader.load_data()
    data = loader.parse_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, all_losses = load_model(args.model, data, device)

    n_confusion = 10000
    confusion = torch.zeros(data.n_genres, data.n_genres).to(device)

    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = data.randomTrainingExample()
        output = evaluate(model, line_tensor.to(device))
        guess, guess_i = data.categoryFromOutput(output)
        category_i = data.genres_list.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(data.n_genres):
        confusion[i] = confusion[i] / confusion[i].sum()

    plot_all_losses(all_losses, save_path=Path(args.model, "all_losses.png").resolve())
    plot_confusion_matrix(
        confusion.to("cpu"),
        data.genres_list,
        save_path=Path(args.model, "confusion.png").resolve(),
    )
