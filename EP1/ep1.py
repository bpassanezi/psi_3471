"""
Exercício programa 1 - PSI3471
Beatriz Soares Passanezi - 10336167
"""

# Importando as bibliotecas a serem usadas
import sys

import cv2
import numpy as np

# Definindo as funções que serão usadas no programa

def acha_vermelho(imagem):
    """
    Retorna a máscara de vermelho da imagem

    (obs: estou achando a máscara em azul devido ao fato de haver placas azuis
    que acabam sendo confundidas com as placas brancas. Isso evita que haja essa confusão)

    Parameters
    ----------
    imagem : matriz
        imagem a ser analisada

    Returns
    -------
    máscara
        máscara onde a imagem é vermelha
    """
    # definindo range do vermelho em HSV
    lower_red_1 = np.array([0,70,60])
    upper_red_1 = np.array([10,255,255])

    lower_red_2 = np.array([170,70,60])
    upper_red_2 = np.array([180,255,255])

    # convertendo a imagem para HSV
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # obtendo as máscaras dos ranges em HSV
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    return np.add(mask1, mask2)


def desenha_circulo(loc, template, imagem):
    """
    Desenha um retangulo na região informada pelo loc

    Parameters
    ----------
    loc : região onde o círculo será desenhado
    template : template da região
    imagem : imagem onde será desenhado o círculo

    Returns
    -------
    imagem: imagem com o círculo desenhado na localização passada
    """
    im_copy = imagem.copy()
    (w, h) = template.shape[:2]

    circle_center = [(int(l + w/2), int(c + h/2)) for (l, c) in zip(*loc[::-1])]

    radius = int(w/2)

    cv2.circle(im_copy, circle_center[0], radius, (0, 255, 0), 5)

    return im_copy


def acha_tamanho(template, imagem_vermelho):
    """
    Encontra o tamanho de template mais apropriado para a imagem

    Parameters
    ----------
    template : template a ser encontrado na imagem
    imagem_vermelho : imagem a ser analisada, já analisada em tons de vermelho

    Returns
    -------
    loc: localização onde o template matching é maior que o threshold
    best_rescale: template do tamanho apropriado
    """
    (w, h) = template.shape[:2]
    max_res = 0
    best_rescale = template
    loc = np.empty(2)

    for rescale_size in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]:
        new_w = int(np.ceil(rescale_size*w))
        new_h = int(np.ceil(rescale_size*h))

        resized_template = cv2.resize(template, (new_w, new_h))

        res = cv2.matchTemplate(imagem_vermelho, resized_template, cv2.TM_CCOEFF_NORMED)

        if res.max() > max_res:
            max_res = res.max()
            loc = np.where(res == max_res)
            best_rescale = resized_template

    return loc, best_rescale


def acha_placa(template, imagem):
    """
    Encontra a placa do template na imagem

    Parameters
    ----------
    template : template de placa a ser utilizado
    imagem : imagem a ser analisada

    Returns
    -------
    imagem: imagem com a placa destacada por um retangulo verde
    """
    imagem_vermelho = acha_vermelho(imagem)

    print("Procurando o tamanho de template que melhor se adequa a imagem...")
    loc, best_rescale = acha_tamanho(template, imagem_vermelho)

    imagem = desenha_circulo(loc, best_rescale, imagem)

    return imagem


# Verificando se os parâmetros passados para a função são adequados
if (len(sys.argv) != 3):
    raise Exception("""Número errado de argumentos passados para o programa...
A chamada esperada é
`python ep1.py <imagem_de_entrada> <imagem_de_saida>`""")

imagem_path = sys.argv[1]
out_path = sys.argv[2]

imagem = cv2.imread(imagem_path, cv2.IMREAD_COLOR)

# Verificando se a imagem passada existe
try:
    imagem.shape
except :
    raise ValueError("A imagem passada não existe")

# Chamando as funções de identificação da placa
template = cv2.imread("template_1.jpg", cv2.IMREAD_GRAYSCALE)

print("Iniciando o processamento da imagem...")
imagem_com_placa = acha_placa(template, imagem)
print("Sua placa foi encontrada!")

print(f"Salvando a imagem com placa em {out_path}")
cv2.imwrite(out_path, imagem_com_placa)
print("Imagem salva!")

