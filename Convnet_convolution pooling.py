from PIL import Image
from math import ceil, sqrt
from math import floor

import matplotlib.pyplot as plt
import numpy as np
import timeit

# Функция свертки с использованием циклов
def convolution(a, b):
    # Определим разницу в размерах между изображением и фильтром
    h_diff = a.shape[0] - b.shape[0]
    w_diff = a.shape[1] - b.shape[1]
    
    # Если размеры не совпадают, добавляем padding к фильтру
    if h_diff > 0 or w_diff > 0:
        b_padded = np.pad(b, [(h_diff//2, h_diff-h_diff//2), (w_diff//2, w_diff-w_diff//2)], mode='constant', constant_values=0)
    else:
        b_padded = b
    
    # Теперь выполняем свертку с новым фильтром
    sum = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            sum += a[i][j] * b_padded[i][j]
    return sum

# Эффективная функция свертки на NumPy
def convolve_numpy(image, kernel):
    """
    Выполняет двумерную свертку изображения с ядром.
    
    Параметры:
    image : ndarray
        Входное изображение.
    kernel : ndarray
        Ядро свертки.
        
    Возвращает:
    result : ndarray
        Результат свертки.
    """
    # Получаем размеры изображения и ядра
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Размер окна вокруг центрального элемента ядра
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Паддинг изображения для сохранения размера
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    # Инициализация результата
    result = np.zeros_like(image)
    
    # Выполнение свертки
    for i in range(image_height):
        for j in range(image_width):
            # Свертка в текущей позиции
            window = padded_image[i:i + kernel_height, j:j + kernel_width]
            result[i, j] = np.sum(window * kernel)
    
    return result

# Основная часть программы
# Открываем изображение и конвертируем его в RGB формат (игнорируем альфа-канал)
img = Image.open('C:\\Users\\TEMP.LAPTOP-EM0D1PRH\\Desktop\\Magistracy\\Artificial Intelligence\\convnet_picture.png').convert('RGB')
pixels = img.load()

# Копия оригинального изображения для применения фильтра
img_convolved = img.copy()
pixels2 = img_convolved.load()

# Пример фильтра
filter = np.array([
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, 4, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1]
])

# Вычисляем сумму всех элементов фильтра
div = 0
for i in range(len(filter)):
    for j in range(len(filter[0])):
        div += filter[i][j]
if div == 0:
    div = 1

# Применяем фильтр к каждому пикселю изображения
for i in range(max(0, floor(len(filter)/2)), min(img.width, img.width - floor(len(filter)/2))):
    for j in range(max(0, floor(len(filter)/2)), min(img.height, img.height - floor(len(filter)/2))):
        matrR = []
        matrG = []
        matrB = []
        for n in range(-floor(len(filter)/2), ceil(len(filter)/2)):
            rowR = []
            rowG = []
            rowB = []
            for m in range(-floor(len(filter)/2), ceil(len(filter)/2)):
                try:
                    # Извлекаем значения R, G, B из текущего пикселя
                    r, g, b = pixels[i + n, j + m]
                    rowR.append(r)
                    rowG.append(g)
                    rowB.append(b)
                except IndexError:
                    # Если индекс выходит за границу изображения, заполняем нулями
                    rowR.append(0)
                    rowG.append(0)
                    rowB.append(0)
                except ValueError:
                    # Если возникает ошибка при извлечении пикселя, сообщаем об этом
                    print(f"Error at position ({i+n}, {j+m}): Got {len(pixels[i + n, j + m])} values instead of 3.")
                    # Инициализируем значения по умолчанию
                    r, g, b = 0, 0, 0
                    rowR.append(r)
                    rowG.append(g)
                    rowB.append(b)
            matrR.append(rowR)
            matrG.append(rowG)
            matrB.append(rowB)

        # Преобразуем matrR в NumPy-массив перед передачей в convolution
        matrR_np = np.array(matrR)
        matrG_np = np.array(matrG)
        matrB_np = np.array(matrB)

        # Применяем свертку к матрицам R, G, B
        r = np.clip(round(convolution(matrR_np, filter) / div), 0, 255)
        g = np.clip(round(convolution(matrG_np, filter) / div), 0, 255)
        b = np.clip(round(convolution(matrB_np, filter) / div), 0, 255)

        # Сохраняем обработанный пиксель
        pixels2[i, j] = (int(r), int(g), int(b))

# Отображаем исходное изображение и результат после применения фильтра
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img)
ax[0].set_title("Original Image")
ax[1].imshow(img_convolved)
ax[1].set_title("Filtered Image")
plt.tight_layout()
plt.show()

# Бенчмаркинг
# Генерация случайного изображения и фильтра
image = np.random.randn(500, 500)
kernel = np.random.randn(3, 3)

# Измерение времени выполнения оригинальной функции свертки
start_time = timeit.default_timer()
convolution(np.array(image), np.array(kernel))
original_time = timeit.default_timer() - start_time

# Измерение времени выполнения функции свертки на NumPy
start_time = timeit.default_timer()
convolve_numpy(image, kernel)
numpy_time = timeit.default_timer() - start_time

# Вывод разницы во времени
print(f"Время выполнения оригинальной функции: {original_time:.4f} секунд")
print(f"Время выполнения функции на NumPy: {numpy_time:.4f} секунд")