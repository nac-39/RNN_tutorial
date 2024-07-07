import torch
import os
from pathlib import Path
from argparse import ArgumentParser
from rnn_excersise.model import RNN
from rnn_excersise.data import DataLoader


def evaluate(trained_rnn, line_tensor):
    hidden = trained_rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = trained_rnn(line_tensor[i], hidden)

    return output


def predict(input_line, trained_rnn, data, n_predictions=3):
    print("\n> %s" % input_line)
    with torch.no_grad():
        output = evaluate(
            trained_rnn, data.lineToTensor(input_line).to(trained_rnn.device)
        )

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print("(%.2f) %s" % (value, data.genres_list[category_index]))
            predictions.append([value, data.genres_list[category_index]])


def load_model(model_path, data, device):
    rnn_path = Path(model_path, "rnn.pth").resolve()
    if not rnn_path.exists():
        raise FileNotFoundError(f"Model not found in {rnn_path}")

    all_losses_path = Path(model_path, "all_losses.pth").resolve()
    if not all_losses_path.exists():
        raise FileNotFoundError(f"all_losses not found in {all_losses_path}")

    model = RNN(data.n_letters, 128, data.n_genres, num_layers=3, device=device).to(
        device
    )
    model.load_state_dict(torch.load(rnn_path))

    # all_lossesを保存
    all_losses = torch.load(all_losses_path)
    return model, all_losses


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_line", type=str, help="Input line to predict")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model directory"
    )
    args = parser.parse_args()
    loader = DataLoader("data", "anime-dataset-2023.csv", "English name")
    loader.load_data()
    data = loader.parse_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, all_losses = load_model(args.model, data, device)
    predict(args.input_line, model, data)
