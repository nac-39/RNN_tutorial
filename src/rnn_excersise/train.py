import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rich.progress import track
from rnn_excersise.model import RNN


class Train:
    print_every = 5000
    plot_every = 1000
    model_path = "models"

    def __init__(self, data, device, logger, n_hidden=128, learning_rate=0.001) -> None:
        self.rnn = RNN(data.n_letters, n_hidden, data.n_genres, device).to(device)
        self.criterion = nn.NLLLoss()
        self.device = device
        self.optimizer = optim.SGD(self.rnn.parameters(), lr=learning_rate)
        self.data = data
        self.logger = logger

    # オプティマイザを定義（例：SGDオプティマイザ）

    def train(self, category_tensor, line_tensor, max_norm):
        hidden = self.rnn.initHidden()

        self.rnn.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, hidden = self.rnn(line_tensor[i], hidden)
        category_tensor = (
            category_tensor.view(1).type(torch.LongTensor).to(self.device)
        )  # バッチサイズ1の整数型テンソルに変換

        loss = self.criterion(output, category_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.rnn.parameters(), max_norm=max_norm
        )  # max_normの値を調整

        # Add parameters' gradients to their values, multiplied by learning rate
        # for p in self.rnn.parameters():
        #     if p is None:
        #         continue
        #     elif not isinstance(p.data, Parameter):
        #         continue
        #     else:
        #         p.data.add_(p.grad.data, alpha=-learning_rate)
        # 勾配がNoneでないことを確認するデバッグ出力
        # for p in self.rnn.parameters():
        #     if p.grad is None:
        #         print(f"Parameter {p} has no gradient.")
        #     else:
        #         print(f"Parameter {p} gradient size: {p.grad.size()}")

        # オプティマイザでパラメータを更新する
        self.optimizer.step()

        return output, loss.item()

    def proceed(self, retrain=False, n_iters=500000, max_norm=5.0):
        current_loss = 0
        all_losses = []

        def timeSince(since):
            now = time.time()
            s = now - since
            m = math.floor(s / 60)
            s -= m * 60
            return "%dm %ds" % (m, s)

        start = time.time()

        # 訓練済みのモデルがあればそれを返す
        if os.path.exists("rnn.pth") and not retrain:
            self.logger.info("model exists")
            # モデルを保存
            model = RNN(self.data.n_letters, self.data.n_hidden, self.data.n_genres).to(
                self.device
            )
            model.load_state_dict(torch.load("rnn.pth"))

            # all_lossesを保存
            all_losses = torch.load("all_losses.pth")
            return model, all_losses
        for iter in track(range(1, n_iters + 1), description="Training model"):
            category, line, category_tensor, line_tensor = (
                self.data.randomTrainingExample()
            )

            output, loss = self.train(category_tensor, line_tensor, max_norm)
            current_loss += loss
            if np.isnan(loss):
                self.logger.error("Loss is NaN.")
                self.logger.error(f"category = {category}")
                self.logger.error(f"line = {line}")
                self.logger.error(f"category_tensor = {category_tensor}")
                self.logger.error(f"line_tensor = {line_tensor}")
                raise ValueError("Loss is NaN.")

            # Print iter number, loss, name and guess
            if iter % self.print_every == 0:
                guess, guess_i = self.data.categoryFromOutput(output)
                correct = "✓" if guess == category else "✗ (%s)" % category
                average_loss = current_loss / self.plot_every
                self.logger.info(
                    f"{iter} {iter / n_iters * 100}% ({timeSince(start)}) {average_loss:.4f} {line} / {guess} {correct}"
                )

            # Add current loss avg to list of losses
            if iter % self.plot_every == 0:
                all_losses.append(current_loss / self.plot_every)
                current_loss = 0
        time_strf = time.strftime("%Y%m%d%H%M%S")
        os.makedirs(f"{self.model_path}/{time_strf}", exist_ok=True)
        # 作成したモデルを保存
        torch.save(self.rnn.state_dict(), f"{self.model_path}/{time_strf}/rnn.pth")
        # all_lossesを保存
        torch.save(all_losses, f"{self.model_path}/{time_strf}/all_losses.pth")
        return self.rnn, all_losses
