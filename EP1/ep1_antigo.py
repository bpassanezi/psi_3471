"""
Exercício programa 1 - PSI3471
Beatriz Soares Passanezi - 10336167
"""

# Importando as bibliotecas a serem usadas
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Definindo as funções que serão usadas no programa

def mostra_imagem_rgb(imagem, title = None):
    """
    Recebe uma imagem BGR, transforma em RGB
    e mostra a imagem usando a biblioteca
    matplotlib

    Parameters
    ----------
    imagem : matriz
        Imagem a ser exibida
    """
    fig = plt.figure(figsize = (10, 10))
    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    plt.title(title)
    plt.imshow(imagem_rgb)
    plt.show()

def acha_vermelho_e_azul(imagem):
    """
    Retorna a máscara de vermelho e azul da imagem

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

    # definindo o range do azul em HSV
    lower_blue = np.array([110,70,60])
    upper_blue = np.array([130,255,255])

    # convertendo a imagem para HSV
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # obtendo as máscaras dos ranges em HSV
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask3 = cv2.inRange(hsv, lower_blue, upper_blue)

    return np.add(np.add(mask1, mask2), mask3)

def desenha_retangulo(loc, template, imagem):
    """
    Desenha um retangulo ao redor da região

    Parameters
    ----------
    loc : região onde o retangulo será desenhado
    template : template da região
    imagem : imagem onde será desenhado o retangulo

    Returns
    -------
    imagem: imagem com o retangulo desenhado na localização passada
    """
    im_copy = imagem.copy()
    (w, h) = template.shape[::-1]

    for (l, c) in zip(*loc[::-1]):
        cv2.rectangle(im_copy, (l, c), ((l + w), (c + h)), (0, 255, 0), 2)

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
    resized_template: template do tamanho apropriado
    """
    w, h = template.shape
    for rescale_size in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]:
        for threshold in [0.8, 0.7, 0.6, 0.5]:
            new_w = int(np.ceil(rescale_size*w))
            new_h = int(np.ceil(rescale_size*h))

            resized_template = cv2.resize(template, (new_w, new_h))

            res = cv2.matchTemplate(imagem_vermelho, resized_template, cv2.TM_CCOEFF_NORMED)

            loc = np.where(res >= threshold)

            if(len(loc[0]) != 0):
                return loc, resized_template

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
    imagem_vermelho = acha_vermelho_e_azul(imagem)

    plt.imshow(imagem_vermelho)
    plt.show()

    loc, resized_template = acha_tamanho(template, imagem_vermelho)
    imagem = desenha_retangulo(loc, resized_template, imagem)

    return imagem

# Chamando as funções definidas
template = cv2.imread("template_1.jpg", cv2.IMREAD_GRAYSCALE)
imagem_path = sys.argv[1]
# threshold = float(sys.argv[2])
imagem = cv2.imread(imagem_path, cv2.IMREAD_COLOR)

mostra_imagem_rgb(imagem, "Imagem original")

imagem_com_placa = acha_placa(template, imagem)
mostra_imagem_rgb(imagem_com_placa, "Imagem com a placa destacada")