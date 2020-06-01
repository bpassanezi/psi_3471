import cv2
import matplotlib.pyplot as plt

"""
Código desenvolvido por: Beatriz Soares Passanezi - 10336167

Lição de casa: Escreva um programa que elimina o ruído branco nas regiões pretas da imagem mickey.bmp.
"""

def elimina_ruido(imagem):
    im_copy = imagem.copy()
    for l in range(1, im_copy.shape[0]-1):
        for c in range(1, im_copy.shape[1]-1):
            # pixel branco
            if im_copy[l, c] == 255:
                soma_pretos = sum([imagem[l+i][c+j] == 0 for (i, j) in [(0, 1), (0, -1), (-1, 0), (0, -1)]])
                if soma_pretos == 4:
                    im_copy[l, c] = 0

    return im_copy

# Lendo a imagem com ruído
mickey_ruido = cv2.imread("basico/mickey.bmp", cv2.IMREAD_GRAYSCALE)

# Eliminando ruído com a função definida acima
mickey = elimina_ruido(mickey_ruido)

# Exibindo as imagens
fig, ax = plt.subplots(1, 2, figsize = (10, 10))

ax[0].set_title("Mickey com ruído")
ax[0].imshow(mickey_ruido, cmap="gray")

ax[1].set_title("Mickey sem ruído")
ax[1].imshow(mickey, cmap="gray")

plt.show()