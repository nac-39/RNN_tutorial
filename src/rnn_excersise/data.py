import zipfile
import os
from pathlib import Path
import random
import string
import torch

import pandas as pd


class DataLoader:
    def __init__(self, dataset_path, dataset_name, learning_column):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.learning_column = learning_column

    def load_data(self):
        # すでにデータセットを格納するディレクトリがあるかどうか確認
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        # データを解凍
        if not os.path.exists(
            Path(self.dataset_path, "anime-dataset-2023.csv").resolve()
        ):
            with zipfile.ZipFile(
                Path(self.dataset_path, "myanimelist-dataset.zip").resolve(), "r"
            ) as zip_ref:
                zip_ref.extractall(self.dataset_path)

    def parse_data(self, device="cpu"):
        anime = pd.read_csv(Path(self.dataset_path, self.dataset_name).resolve())
        anime.dropna()

        def random_index(list_length):
            return random.randint(0, list_length - 1)

        # データを整形する
        anime = anime[[self.learning_column, "Genres"]]

        # ジャンルを最初の１つだけにする
        anime["Genres"] = anime["Genres"].apply(
            lambda x: x.split(", ")[random_index(len(x.split(", ")))]
        )

        # UNKNWONが含まれるカラムを表示
        anime[anime["Genres"].str.contains("UNKNOWN")]

        # self.learning_columnに UNKNOWNを含む行を削除
        anime = anime[~anime[self.learning_column].str.contains("UNKNOWN")]

        # データの数などの情報を取得
        genres_list = list(anime.Genres.str.split(", ").explode().unique())

        data = Data(genres_list, anime, self.learning_column, device)
        return data


class Data:
    def __init__(self, genres_list, anime, learn_column, device):
        self.genres_list = genres_list
        self.anime = anime
        self.n_genres = len(genres_list)
        self.device = device
        self.learn_column = learn_column

    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)

    # Find letter index from all_letters, e.g. "a" = 0
    @classmethod
    def letterToIndex(cls, letter):
        return cls.all_letters.find(letter)

    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    def letterToTensor(self, letter):
        tensor = torch.zeros(1, self.n_letters).to(self.device)
        tensor[0][self.letterToIndex(letter)] = 1
        return tensor

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def lineToTensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters).to(self.device)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor

    @staticmethod
    def randomChoice(list):
        return list[random.randint(0, len(list) - 1)]

    def randomTrainingExample(self):
        genre = self.randomChoice(self.genres_list)
        anime_titles = self.anime[self.anime["Genres"].str.contains(genre, case=False)][
            self.learn_column
        ]

        line = self.randomChoice(anime_titles.to_list())
        genre_tensor = torch.tensor(self.genres_list.index(genre), dtype=torch.long).to(
            self.device
        )
        line_tensor = self.lineToTensor(line).to(self.device)
        return genre, line, genre_tensor, line_tensor

    def categoryFromOutput(self, output):
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return self.genres_list[category_i], category_i
