# rnn-tutorial

## チュートリアル
[PyTorchチュートリアル（日本語翻訳版）](https://yutaroogawa.github.io/pytorch_tutorials_jp/)のコードがそのまま入っています。

## 演習
チュートリアルを元に、アニメのタイトルからジャンルを推測することを試みたものです。
データセットは[Anime Dataset 2023](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)を使っています。
`src/rnn_excersise/notebooks/anime.ipynb`が最新のコードを実行できるノートブックになっています。

以下のコマンドでも実行することができます。`{model_id}`は訓練したモデルを保存したディレクトリ名です。

```bash

rye run python main.py \
    --retrain \
    --n_iters 100000 \
    --learning_rate 0.0001 \
    --num_layers 3 \
    --max_norm 5.0

rye run python plot.py 10000 \
    --model models/{model_id}

```

## VSCodeでjupyter notebookを開く方法

1. 仮想環境に入ったあと、以下のコマンドを実行する。
```bash
$ jupyter notebook --ip=* --no-browser
```

2. このようなURLが表示されるので、httpの方をコピーする。
```
...
   To access the server, open this file in a browser:
        file:///home/s.nanako/.local/share/jupyter/runtime/jpserver-19578-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/tree?token=8f3c1ae38ff3369c811278c60a7ebcd6867a3c8bd9404053
        http://127.0.0.1:8888/tree?token=8f3c1ae38ff3369c811278c60a7ebcd6867a3c8bd9404053
...
```

3. .ipynbファイルを開いたあと、右上の「Select Jupyter Server」をクリックし、先ほどコピーしたURLを入力する。



