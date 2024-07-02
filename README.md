# rnn-tutorial

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



