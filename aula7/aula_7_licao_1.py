"""
Lição de casa 1
A imagem horver.png contém traços horizontais e verticais. Escreva um programa
que lê horver.png e separa os traços horizontais dos verticais usando morfologia matemática,
gerando hor.png e ver.png respectivamente com traços horizontais e verticais somente.

Código desenvolvido por: Beatriz Soares Passanezi - 10336167
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

horver = cv2.imread("horver.png")

kernel_hor = np.array([[1, 1, 1, 1, 1]],np.uint8)  # kernel para detectar traços horizontais
kernel_ver = np.array([[1],[1], [1], [1]],np.uint8)  # kernel para detectar traços verticais

# Aplicando abertura na imagem usando os respectivos kernels
horver_hor = cv2.morphologyEx(horver, cv2.MORPH_OPEN, kernel_hor)
horver_ver = cv2.morphologyEx(horver, cv2.MORPH_OPEN, kernel_ver)

# Salvando as imagens
cv2.imwrite("hor.png", horver_hor)
cv2.imwrite("ver.png", horver_ver)

# Exibindo as imagens criadas
fig, ax = plt.subplots(1, 3, figsize=(12, 6))

ax[0].imshow(horver)
ax[0].set_title("Imagem original")

ax[1].imshow(horver_hor)
ax[1].set_title("Imagem com traços horizontais")

ax[2].imshow(horver_ver)
ax[2].set_title("Imagem com traços verticais")

plt.show()