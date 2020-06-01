"""
Lição de casa 1: Escreva um programa que usa a função medianBlur do OpenCV para aplicar
filtro mediana na imagem de entrada. Aplique o programa resultante na imagem ruido.png e
verifique que realmente elimina ruído.
"""

# Lendo a imagem original
img = cv2.imread("filtros/ruido.png")

# Passando um filtro mediana pela imagem com tamanho 3x3
median = cv2.medianBlur(img, 3)

# Exibindo as imagens
fig, ax = plt.subplots(1, 2, figsize = (10, 10))

ax[0].set_title("Imagem original")
ax[0].imshow(img)

ax[1].set_title("Imagem após passar pelo filtro mediana")
ax[1].imshow(median)

plt.show()