import cv2
import queue
import matplotlib.pyplot as plt

"""
Código desenvolvido por Beatriz Passanezi - 10336167

Lição de casa: Escreva um programa que lê uma imagem binária e imprime o número de
componentes conexos da imagem. Rodando o seu programa para imagem "letras.bmp", deve
retornar 31. Aqui, estou chamando de componente conexo o conjunto de pixels pretos grudados entre si.
"""

# Definindo a função que pinta componentes pretos em vermelho dada a semente (li, ci)
def pintaPretoVermelho(a,li,ci):
    b=a.copy()
    q=queue.Queue()
    q.put(li)
    q.put(ci)
    while not q.empty():
        l=q.get()
        c=q.get()
        if all(b[l,c,:]==[0,0,0]):
            b[l,c]=[255,0,0]
            q.put(l-1); q.put(c)
            q.put(l+1); q.put(c)
            q.put(l); q.put(c+1)
            q.put(l); q.put(c-1)
    return b

# Definindo a função que conta o número de componentes conexos
def contaConexos(a):
    b = a.copy()
    q = queue.Queue()

    q.put(0)
    q.put(0)

    count = 0

    for l in range(b.shape[0]):
        for c in range(b.shape[1]):
            if all(b[l,c,:] == [0,0,0]): # se nao for branco
                count = count + 1
                b = pintaPretoVermelho(b, l, c)

    return count, b

# Lendo a imagem letras
letras = cv2.imread('letras.bmp',cv2.IMREAD_COLOR)

# Contando o número de componentes conexos
count, letras_conexos = contaConexos(letras)

print(f"O número de componentes conexos na imagem é {count}")