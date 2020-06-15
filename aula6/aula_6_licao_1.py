import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Lição de casa: Modifique o programa 9 para obter programa que converte quadrado2.png em
quadrado2b.png

Código desenvolvido por Beatriz Passanezi - 10336167
"""


# Lendo imagem de entrada
entrada = cv2.imread("quadrado2.png", 0)

# Definição dos pontos correspondentes a imagem de origem e de saída
src = np.array([(237, 155), (640, 205), (0, 260), (449, 323)], dtype = "float32")
dst = np.array([(1, 0), (620, 0), (1, 450), (620, 450)], dtype = "float32")

# Fazendo a conversão das imagens
m = cv2.getPerspectiveTransform(src, dst)
saida = cv2.warpPerspective(entrada,m,(620, 450))

# Salvando imagem de saída
cv2.imwrite("quadrado2b.png", saida)

# Exibindo as imagens
fig, ax = plt.subplots(1, 2, figsize=(15, 10))

ax[0].imshow(entrada, cmap="gray")
ax[0].set_title("Imagem original")

ax[1].imshow(saida, cmap="gray")
ax[1].set_title("Imagem corrigida")

plt.show()