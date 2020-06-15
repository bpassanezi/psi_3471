import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Lição de casa 2
Adapte o algoritmo 9 para ler a imagem fundoolhog.jpg, e gerar 2 imagens de
saída: sem_mancha_clara.jpg e sem_mancha_escura.jpg. Procure apagar somente as manchas
especificadas.
"""

# https://codereview.stackexchange.com/questions/231225/create-a-structuringselement-in-the-form-of-a-line-with-a-certain-degree-and-len

def bresenham(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))

    return points

def strel_line(length, degrees):
    if length >= 1:
        theta = degrees * np.pi / 180
        x = round((length - 1) / 2 * np.cos(theta))
        y = -round((length - 1) / 2 * np.sin(theta))
        points = bresenham(-x, -y, x, y)
        points_x = [point[0] for point in points]
        points_y = [point[1] for point in points]
        n_rows = int(2 * max([abs(point_y) for point_y in points_y]) + 1)
        n_columns = int(2 * max([abs(point_x) for point_x in points_x]) + 1)
        strel = np.zeros((n_rows, n_columns))
        rows = ([point_y + max([abs(point_y) for point_y in points_y]) for point_y in points_y])
        columns = ([point_x + max([abs(point_x) for point_x in points_x]) for point_x in points_x])
        idx = []
        for x in zip(rows, columns):
            idx.append(np.ravel_multi_index((int(x[0]), int(x[1])), (n_rows, n_columns)))
        strel.reshape(-1)[idx] = 1

    return strel

def max_arrays(a, b):
    c = np.empty(a.shape)
    (num_col, num_l) = a.shape[:2]
    for i in range(num_col):
        for j in range(num_l):
            c[i, j] = max(a[i, j], b[i, j])
    return c

def min_arrays(a, b):
    c = np.empty(a.shape)
    (num_col, num_l) = a.shape[:2]
    for i in range(num_col):
        for j in range(num_l):
            c[i, j] = min(a[i, j], b[i, j])
    return c


def acha_mancha(imagem, num_it):
    c = np.empty(imagem.shape)

    for deg in range(1, 181, 10):
        e = strel_line(50, deg).astype(np.uint8)
        b = cv2.morphologyEx(imagem, cv2.MORPH_OPEN, e)
        c = max_arrays(b, c)

    e33 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))*255
    for i in range(num_it):
        print(f"{i/num_it} concluido")
        c = min_arrays(cv2.dilate(c, e33), imagem)
    return c

fundo_olho = cv2.imread("fundoolhog.jpg", 0)
fundo_olho_neg = cv2.bitwise_not(fundo_olho)

print("Calculando imagem sem mancha clara")
sem_mancha_clara = acha_mancha(fundo_olho, 200)
print("Calculando imagem sem mancha escura")
sem_mancha_escura = acha_mancha(fundo_olho_neg, 200)

plt.imshow(sem_mancha_clara, cmap="gray")
plt.title("Sem mancha clara")
plt.show()
plt.imshow(abs(255-sem_mancha_escura), cmap="gray")
plt.title("Sem mancha escura")
plt.show()