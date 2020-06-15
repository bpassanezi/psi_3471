"""
Lição de casa
Modifique os programas 4 ou 5 para obter taxa de erro de teste menor que 2% (sem
usar rede neural convolucional). Algumas alterações possíveis são:
(a) Acrescentar ou eliminar camadas.
(b) Mudar o número de neurônios das camadas.
(c) Mudar função de ativação.
(d) Mudar o
otimizador e/ou os seus parâmetros.
(e) Mudar o tamanho do batch.
(f) Mudar o número de épocas.
(g) Eliminar linhas/colunas brancas das imagens de entrada.
(h) Redimensionar as imagens de entrada.
(i) Aumentar artificialmente os dados de treinamento, criando versões distorcidas das imagens.

Código desenvolvido por Beatriz Soares Passanezi - 10336167
"""

import tensorflow.keras as keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import sys

"""
As modificações feitas foram:
- alterar o número de epochs
- alterar o tamanho de batch
- alterar o número de neuronios na camada
- usar o ReduceLROnPlateau para reduzir a learning rate ao chegar em um plateau de perda
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = (255 - x_train)/255.0
x_test = (255 - x_test)/255.0

nclasses = 10
y_train = keras.utils.to_categorical(y_train, nclasses)
y_test = keras.utils.to_categorical(y_test, nclasses)

num_linhas = x_train.shape[1]
num_cols = x_train.shape[2]

# Criando o modelo
model2 = Sequential()
model2.add(Flatten(input_shape=(num_linhas,num_cols)))
model2.add(Dense(400, activation='relu'))
model2.add(Dense(150, activation='relu'))
model2.add(Dense(nclasses, activation='softmax'))

opt = optimizers.Adam()
model2.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=2, min_lr=0.0001, verbose=True)

# Treinando o modelo
model2.fit(x_train, y_train,
           batch_size=500,
           epochs=150,
           verbose=True,
           validation_data=(x_test, y_test),
           callbacks=[reduce_lr])

# Avaliando o modelo
score = model2.evaluate(x_test, y_test, verbose=False)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]*100}%')
print(f'Test error: {100 - score[1]*100}%')

# Salvando o modelo
model2.save('licao_1.h5')
"""
O modelo salvo teve os resultados

Test loss: 0.0801842857906545
Test accuracy: 98.14000129699707%
Test error: 1.8599987030029297%

"""