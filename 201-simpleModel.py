#!/usr/bin/env
# coding:utf-8

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers

#モデル定義
model = L.Linear(1,1)
optimizer = optimizers.SGD()
optimizer.setup(model)

#学習させる回数
times = 300

#入力ベクトル
x = Variable(np.array([[1],[2],[7]],dtype=np.float32))

#正解ベクトル
t = Variable(np.array([[2],[4],[14]],dtype=np.float32))

#学習ループ
for i in range(0,times):
    #こうはいを初期化
    optimizer.zero_grads()

    #ここにモデルを予測させている
    y = model(x)

    #モデルが出した答えを表示
    print(y.data)

    #損失を計算する
    loss = F.mean_squared_error(y,t)

    #逆伝播する
    loss.backward()

    #optimizerを更新する
    optimizer.update()

print "result"
x = Variable(np.array([[3],[4],[5]],dtype=np.float32))
y = model(x)
print(y.data)
