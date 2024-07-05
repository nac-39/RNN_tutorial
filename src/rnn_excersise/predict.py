import torch


def evaluate(trained_rnn, line_tensor):
    hidden = trained_rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = trained_rnn(line_tensor[i], hidden)

    return output


def predict(input_line, trained_rnn, data, n_predictions=3):
    print("\n> %s" % input_line)
    with torch.no_grad():
        output = evaluate(trained_rnn, data.lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print("(%.2f) %s" % (value, data.genres_list[category_index]))
            predictions.append([value, data.genres_list[category_index]])
