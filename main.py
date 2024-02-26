import cv2
import numpy as np
import time

# Считываем время
start_time = time.time()

# Считываем изображение
image = cv2.imread('test_img.bmp')

# Переводим изображение в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применяем пороговую обработку для выделения контуров
ret, threshold = cv2.threshold(gray_image, 40, 255, 0)

# Находим контуры на изображении
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Выбираем наибольший контур
largest_contour = max(contours, key=cv2.contourArea)

# Аппроксимируем контур для сглаживания линий
epsilon = 0.015 * cv2.arcLength(largest_contour, True)
epsilon_2 = 0.005 * cv2.arcLength(largest_contour, True)
approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
approx_contour_2 = cv2.approxPolyDP(largest_contour, epsilon_2, True)

# Находим самый длинный отрезок контура
max_segment_length = 0
max_segment_index = 0
for i in range(len(approx_contour) - 1):
    segment_length = np.linalg.norm(approx_contour[i] - approx_contour[i+1])
    if segment_length > max_segment_length:
        max_segment_length = segment_length
        max_segment_index = i

# Используем конечные точки самого длинного отрезка как начальные точки среза
start_point1 = tuple(approx_contour[max_segment_index][0])
start_point2 = tuple(approx_contour[max_segment_index+1][0])

# Найдем длину среза
chord_length = np.sqrt((start_point2[0] - start_point1[0])**2 + (start_point2[1] - start_point1[1])**2)

# Определяем общую длину контура
perimeter = cv2.arcLength(approx_contour_2, True)

# Определяем процентное соотношение между длиной среза и общим периметром
percent_chord_to_perimeter = (chord_length / perimeter) * 100

# Считываем время
end_time = time.time()

# Отображаем контур объекта
cv2.drawContours(image, [approx_contour_2], -1, (0, 255, 0), 2)

# Отрисовываем срез на изображении
cv2.line(image, start_point1, start_point2, (0, 0, 255), 2)


print("Общая длина периметра:", perimeter)
print("Длина среза:", chord_length)
print("Процентное соотношение между длиной среза и общим периметром:", percent_chord_to_perimeter)
print(f"Время выполнения алгоритма: {round(end_time - start_time, 5)} секунд")

cv2.imshow('Contour with Chord', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
