import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

"""
Código desenvolvido por: Beatriz Soares Passanezi - 10336167
Lição de casa: Quebre uma imagem colorida em componentes RGB, como mostra a figura 7.
"""

# Lendo a imagem original
flor = cv2.imread("basico/flor.jpg")

# Transformando imagem em RGB
flor_rgb = cv2.cvtColor(flor, cv2.COLOR_BGR2RGB)

# Separando os canais Red, Green, Blue
red = flor_rgb[:, :, 0]
green = flor_rgb[:, :, 1]
blue = flor_rgb[:, :, 2]

# Exibindo as imagens
norm = Normalize(0, 255)

fig, ax = plt.subplots(2, 2, figsize = (10, 10))

ax[0][0].set_title("Imagem original")
ax[0][0].imshow(flor_rgb)

ax[0][1].set_title("Vermelho")
ax[0][1].imshow(red, norm=norm, cmap="Reds_r")

ax[1][0].set_title("Azul")
ax[1][0].imshow(blue, norm=norm, cmap="Blues_r")

ax[1][1].set_title("Verde")
ax[1][1].imshow(blue, norm=norm, cmap="Greens_r")

plt.show()