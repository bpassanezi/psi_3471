"""
Lição de casa 2: Altere o programa 7 para calcular gradiente usando núcleo de Sobel. Rode o
programa resultante na imagem fantom.pgm e verifique se as saídas são diferentes daquelas
obtidas usando núcleo de Scharr.
"""

img = cv2.imread("convolucao/fantom.pgm")

sobel_kernel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], np.float32)/3.0
sobel_kernel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], np.float32)/3.0

# Gradiente na direção y
filtro_y = cv2.filter2D(img,-1,sobel_kernel_y)

# Gradiente na direção x
filtro_x = cv2.filter2D(img,-1,sobel_kernel_x)

# Calculando o módulo do gradiente nas duas direções
modulo_ao_quadrado = np.add(np.square(filtro_y), np.square(filtro_x))

# Exibindo as imagens
fig, ax = plt.subplots(2, 2, figsize = (10, 10))

ax[0][0].set_title("Imagem original")
ax[0][0].imshow(img)

ax[0][1].set_title("Módulo ao quadrado do gradiente")
ax[0][1].imshow(modulo_ao_quadrado)

ax[1][0].set_title("Gradiente na direção x")
ax[1][0].imshow(filtro_x)

ax[1][1].set_title("Gradiente na direção y")
ax[1][1].imshow(filtro_y)

plt.show()