"""по варианту формулы 5.3, 5.5 """
import matplotlib.pylab as plt
import numpy as np
import cv2


def calculate_moment(mask, order):
    # функция np.unique возвращает отсортированный массив оставляя в нем
    # только уникальные значения
    # если параметр return_counts=True также возвращает
    # количество вхождений каждого элемента в исходный массив
    # в случае когда параметр axis не указак (по умолчанию None)
    # входной массив сжимается до одной оси
    # фактически получаем гистограмму массива

    unique_mask, hist = np.unique(mask, return_counts=True)
    hist = hist / mask.shape[0] / mask.shape[1]  # нормировка значений к диапазону [0; 1]

    meanz = 0
    for i in range(hist.shape[0]):
        meanz += unique_mask[i] * hist[i]

    moment = 0
    for i in range(hist.shape[0]):
        moment += ((unique_mask[i] - meanz) ** order) * hist[i]
    norm_moment = moment / (255 ** order)  # нормировка 2го момента к диапазону [0; 1]

    return norm_moment


def calculate_r(moment2):
    R = 1 - (1 / (1 + moment2))
    return R


def get_metrics(img, mask_size):
    # mask_size only odd number
    indent = int((mask_size - 1) / 2)
    img_r = np.zeros(img.shape)
    img_m3 = np.zeros(img.shape)
    padimg = np.pad(img, (indent, indent), 'edge')

    # итерация по изображению
    for i in range(indent, img.shape[0] - indent):
        for j in range(indent, img.shape[1] - indent):
            my_mask = padimg[i - indent: i + indent + 1, j - indent: j + indent + 1]
            moment2 = calculate_moment(my_mask, 2)
            moment3 = calculate_moment(my_mask, 3)
            r = calculate_r(moment2)
            img_r[i - indent, j - indent] = r
            img_m3[i - indent, j - indent] = moment3
    return img_r, img_m3


def thresholding(img_r, img_m3, r_th_sm, r_th_per_ro, m3_th_sm, m3_th_per_ro):
    norm_m3 = (img_m3 * (-1)) / np.max((img_m3 * (-1))) * 255
    norm_r = img_r / np.max(img_r) * 255

    # разделяем текстуры на изображении с вычисленным R
    # нормализованный диапазон 0-255
    # поэтому значения для текстур следующие
    # гладкая [0; 85*) -> 30; r_th_sm
    # периодиечская [85; 170*) -> 85; r_th_per_ro
    # грубая [*170; 255] -> 140 r_th_per_ro
    r_texture = np.zeros(norm_r.shape)
    r_texture = np.where(norm_r >= r_th_per_ro, 140, r_texture)
    r_texture = np.where(norm_r < r_th_per_ro, 85, r_texture)
    r_texture = np.where(norm_r < r_th_sm, 30, r_texture)

    # разделяем текстуры на изображении с вычисленным m3
    # нормализованный диапазон 0-255
    # поэтому значения для текстур следующие
    # гладкая [0; 85*) -> 15; m3_th_sm
    # периодиечская [85; 170*) -> 45; m3_th_per_ro
    # грубая [*170; 255] -> 70 m3_th_per_ro
    m3_texture = np.zeros(norm_m3.shape)
    m3_texture = np.where(norm_m3 >= m3_th_per_ro, 70, m3_texture)
    m3_texture = np.where(norm_m3 < m3_th_per_ro, 46, m3_texture)
    m3_texture = np.where(norm_m3 < m3_th_sm, 15, m3_texture)

    # складываем два полученных текстурированных изображения
    sum_texture = r_texture + m3_texture

    return r_texture, m3_texture, sum_texture


def create_marking(img):
    labels = np.zeros(img.shape, dtype=int)
    numb_labl = 1
    connection_table1 = np.zeros((img.shape[1], 2))

    # перебор строк изображения
    for i in range(img.shape[0]):
        # проход по строке в прямом направлении
        connection_table2 = np.zeros((img.shape[1], 2))
        for j in range(img.shape[1]):
            # обработка превого пикселя
            if i == 0 and j == 0:
                labels[i, j] = numb_labl
                connection_table1[j, 0] = labels[i, j]
                numb_labl += 1
                continue
            # обработка первой строки
            elif i == 0:
                if (j - 1 >= 0) and (img[i, j] == img[i, j - 1]):
                    labels[i, j] = labels[i, j - 1]
                    connection_table1[j, 0] = labels[i, j]
                else:
                    labels[i, j] = numb_labl
                    connection_table1[j, 0] = labels[i, j]
                    numb_labl += 1
            # проверка выполняемая для каждого пикселя
            # начинаем с левого и двигаемся по часовой стрелке
            # (j - 1 >= 0) проверка на крайний левый пиксель
            # (j + 1 < img.shape[1]) проверка на крайний правый пиксель
            if i > 0:
                if (j - 1 >= 0) and (img[i, j] == img[i, j - 1]):
                    labels[i, j] = labels[i, j - 1]
                    connection_table2[j, 0] = labels[i, j]
                elif (j - 1 >= 0) and (img[i, j] == img[i - 1, j - 1]):
                    labels[i, j] = labels[i - 1, j - 1]
                    connection_table2[j, 0] = labels[i, j]
                elif (i - 1 >= 0) and img[i, j] == img[i - 1, j]:
                    labels[i, j] = labels[i - 1, j]
                    connection_table2[j, 0] = labels[i, j]
                elif (j + 1 < img.shape[1]) and (img[i, j] == img[i - 1, j + 1]):
                    labels[i, j] = labels[i - 1, j + 1]
                    connection_table2[j, 0] = labels[i, j]
                else:
                    labels[i, j] = numb_labl
                    connection_table2[j, 0] = labels[i, j]
                    numb_labl += 1

                # проверка наличия среди соседей равных значений пикселей при неравных значениях меток
                if (j - 1 >= 0) and (img[i, j] == img[i, j - 1]) and (labels[i, j] != labels[i, j - 1]):
                    connection_table2[j, 1] = labels[i, j - 1]
                elif (j - 1 >= 0) and (img[i, j] == img[i - 1, j - 1]) and (labels[i, j] != labels[i - 1, j - 1]):
                    connection_table2[j, 1] = labels[i - 1, j - 1]
                elif (i - 1 >= 0) and img[i, j] == img[i - 1, j] and (labels[i, j] != labels[i - 1, j]):
                    connection_table2[j, 1] = labels[i - 1, j]
                elif (j + 1 < img.shape[1]) and (img[i, j] == img[i - 1, j + 1]) and (
                        labels[i, j] != labels[i - 1, j + 1]):
                    connection_table2[j, 1] = labels[i - 1, j + 1]

        # проход по строке в обратном напрвлении, выполняется для второй
        # и всех последующих строк
        # то есть мы должны иметь предыдущую строку для того,
        # чтобы этот уточняющий значения меток проход имел смысл
        if i > 0:
            for k in range(img.shape[1] - 1, -1, -1):
                if connection_table2[k, 1] != 0:
                    if k == 8:
                        pass
                    target_value = np.min(connection_table2[k, ])
                    change_value = np.max(connection_table2[k, ])
                    labels[i, k] = target_value
                    if (k + 1 < img.shape[1]) and (labels[i, k + 1] == change_value):
                        labels = np.where(labels == change_value, target_value, labels)
                    if (k + 1 < img.shape[1]) and (labels[i - 1, k + 1] == change_value):
                        labels = np.where(labels == change_value, target_value, labels)
                    if labels[i - 1, k] == change_value:
                        labels = np.where(labels == change_value, target_value, labels)
                    if (k - 1 >= 0) and (labels[i - 1, k - 1] == change_value):
                        labels = np.where(labels == change_value, target_value, labels)
            connection_table2[:, 1] = 0
            connection_table1 = connection_table2

    return np.array(labels, dtype=int)


def remove_small_area(img, labels, threshold):
    for i in range(np.max(labels) + 1):
        counting = np.where(labels == i, True, False)
        size_area = np.count_nonzero(counting)
        if size_area < threshold:
            img = np.where(labels == i, 255, img)
    return img


def apply_mask(img, mask):
    result = np.where(mask <= 45, mask, img)
    return result


# image = cv2.imread('lab_image.jpeg', cv2.IMREAD_GRAYSCALE)
# imgNumb = 1
# for mask_size in [3]:
#     image_r, image_m3 = get_metrics(image, mask_size)
#     for r_th_sm in range(1, 3):
#         for r_th_per_ro in range(3, 10, 5):
#             for m3_th_sm in range(1, 6, 2):
#                 for m3_th_per_ro in range(1, 6, 2):
#                     r_texture, m3_texture, sum_texture = thresholding(image_r,
#                                                                       image_m3,
#                                                                       r_th_sm,
#                                                                       r_th_per_ro,
#                                                                       m3_th_sm * 0.1,
#                                                                       m3_th_per_ro)
#                     r_labels = create_marking(r_texture)
#                     m3_labels = create_marking(m3_texture)
#                     sum_labels = create_marking(sum_texture)
#                     for threshold in range(500, 701, 100):
#                         r_mask = remove_small_area(r_texture, r_labels, threshold)
#                         m3_mask = remove_small_area(m3_texture, m3_labels, threshold)
#                         sum_mask = remove_small_area(sum_texture, sum_labels, threshold)
#
#                         r_result = apply_mask(image, r_mask)
#                         m3_result = apply_mask(image, m3_mask)
#                         sum_result = apply_mask(image, sum_mask)
#
#                         r_out = np.concatenate((r_texture, r_mask, r_result), axis=1)
#                         m3_out = np.concatenate((m3_texture, m3_mask, m3_result), axis=1)
#                         sum_out = np.concatenate((sum_texture, sum_mask, sum_result), axis=1)
#
#                         r_name = '.\\lab_image_3\\' + str(imgNumb) +\
#                                  ' lab_img r_stat mask_size=' + str(mask_size) + \
#                                  ' r_th_sm=' + str(r_th_sm) + ' r_th_per_ro=' + \
#                                  str(r_th_per_ro) + ' threshold=' + \
#                                  str(threshold) + '.jpeg'
#                         cv2.imwrite(r_name, r_out)
#                         print(imgNumb, ' / 324')
#                         imgNumb += 1
#
#                         m3_name = '.\\lab_image_3\\' + str(imgNumb) + \
#                                   ' lab_img m3_stat mask_size=' + str(mask_size) + \
#                                   ' m3_th_sm=' + str(m3_th_sm) + ' m3_th_per_ro=' + \
#                                   str(m3_th_per_ro) + ' threshold=' + \
#                                   str(threshold) + '.jpeg'
#                         cv2.imwrite(m3_name, m3_out)
#                         print(imgNumb, ' / 324')
#                         imgNumb += 1
#
#                         sum_name = '.\\lab_image_3\\' + str(imgNumb) + \
#                                    ' lab_img sum_stat mask_size=' + str(mask_size) + \
#                                    ' r_th_sm=' + str(r_th_sm) + ' r_th_per_ro=' + \
#                                    str(r_th_per_ro) + ' m3_th_sm=' + str(m3_th_sm) + \
#                                    ' m3_th_per_ro=' + str(m3_th_per_ro) + \
#                                    ' threshold=' + str(threshold) + '.jpeg'
#                         cv2.imwrite(sum_name, sum_out)
#                         print(imgNumb, ' / 324')
#                         imgNumb += 1

# тест одного случая
# изменяемые параметры
mask_size = 3  # для функции get_metrics
r_th_sm = 2  # для thresholding
r_th_per_ro = 8
m3_th_sm = 5
m3_th_per_ro = 1
threshold = 600  # для функции remove_small_area

image = cv2.imread('google.jpeg', cv2.IMREAD_GRAYSCALE)
imgNumb = 1
image_r, image_m3 = get_metrics(image, mask_size)
r_texture, m3_texture, sum_texture = thresholding(image_r, image_m3, r_th_sm, r_th_per_ro, m3_th_sm * 0.1, m3_th_per_ro)
r_labels = create_marking(r_texture)
r_mask = remove_small_area(r_texture, r_labels, threshold)
r_result = apply_mask(image, r_mask)
r_out = np.concatenate((r_texture, r_mask, r_result), axis=0)
r_name = '.\\my_image_1\\' + str(imgNumb) + ' my_img r_stat mask_size=' + \
         str(mask_size) + ' r_th_sm=' + str(r_th_sm) + ' r_th_per_ro=' + \
         str(r_th_per_ro) + ' threshold=' + str(threshold) + '.jpeg'
cv2.imwrite(r_name, r_out)
print(imgNumb, ' / 480')
imgNumb += 1

m3_labels = create_marking(m3_texture)
m3_mask = remove_small_area(m3_texture, m3_labels, threshold)
m3_result = apply_mask(image, m3_mask)
m3_out = np.concatenate((m3_texture, m3_mask, m3_result), axis=0)
m3_name = '.\\my_image_1\\' + str(imgNumb) + \
' my_img m3_stat mask_size=' + str(mask_size) + \
' m3_th_sm=' + str(m3_th_sm) + ' m3_th_per_ro=' + \
str(m3_th_per_ro) + ' threshold=' + \
str(threshold) + '.jpeg'
cv2.imwrite(m3_name, m3_out)
print(imgNumb, ' / 324')
imgNumb += 1

sum_labels = create_marking(sum_texture)
sum_mask = remove_small_area(sum_texture, sum_labels, threshold)
sum_result = apply_mask(image, sum_mask)
sum_out = np.concatenate((sum_texture, sum_mask, sum_result), axis=0)
sum_name = '.\\my_image_1\\' + str(imgNumb) + \
' my_img sum_stat mask_size=' + str(mask_size) + \
' r_th_sm=' + str(r_th_sm) + ' r_th_per_ro=' + \
str(r_th_per_ro) + ' m3_th_sm=' + str(m3_th_sm) + \
' m3_th_per_ro=' + str(m3_th_per_ro) + \
' threshold=' + str(threshold) + '.jpeg'
cv2.imwrite(sum_name, sum_out)
print(imgNumb, ' / 324')
imgNumb += 1
