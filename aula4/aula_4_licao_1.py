"""
Lição de casa

Escreva um programa que detecta as 4 ocorrências de urso “q.png” na imagem
a analisar “a.png” gerando uma imagem processada semelhante a “p.png”. Só precisa entregar
o código .cpp ou .py.

Código desenvolvido por: Beatriz Soares Passanezi - 10336167
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Lendo os arquivos da imagem a ser analisada e do template
imagem_base = cv2.imread("a.png")
template = cv2.imread("q.png")

# Aplicando a função de template matching com o modo TM_CCOEFF_NORMED, que é
# invariante a brilho e contraste e pegando o seu módulo pois o template é
# preto, mas há um ursinho branco
res = abs(cv2.matchTemplate(imagem_base,template,cv2.TM_CCOEFF_NORMED))

# Encontrando os valores onde o módulo do template match é maior que 0.8
loc = np.where(res >= 0.8)

# Destacando os pontos de máximo módulo em vermelho
imagem_match = imagem_base.copy()
for l, c in zip(*loc):
    imagem_match[l, c] = (255, 0, 0)
    imagem_match[l + 1, c + 1] = (255, 0, 0)
    imagem_match[l - 1, c - 1] = (255, 0, 0)

# Exibindo os resultados
fig, ax = plt.subplots(1, 2, figsize = (15, 10))

ax[0].imshow(imagem_base)
ax[0].set_title("Imagem base")

ax[1].imshow(imagem_match)
ax[1].set_title("Imagem com os ursos encontrados")

plt.show()