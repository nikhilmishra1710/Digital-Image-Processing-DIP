import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    img = cv2.imread(path)
    initial_matrix_rgb = np.array(img)

    plt.imshow(initial_matrix_rgb)
    plt.show()

    dimensions = img.shape
    print(dimensions)

    print('Height of Image:', dimensions[0])
    print('Width of Image:', dimensions[1])
    print('Number of Dimensions:', dimensions[2])

    return initial_matrix_rgb

def convert_to_grayscale(initial_matrix_rgb):
    row = initial_matrix_rgb.shape[0]
    col = initial_matrix_rgb.shape[1]
    print(row, col)
    initial_matrix_gray = np.zeros((row, col), dtype="int")

    for i in range(row):
        for j in range(col):
            initial_matrix_gray[i, j] = round(
                0.2989 * initial_matrix_rgb[i][j][0] +
                0.5870 * initial_matrix_rgb[i][j][1] +
                0.1140 * initial_matrix_rgb[i][j][2]
            )

    plt.imshow(initial_matrix_gray, cmap="gray")
    plt.show()

    return initial_matrix_gray

def calculate_cdf(initial_matrix_gray):
    values = np.unique(initial_matrix_gray)
    print("Unique pixel values:", values)

    frequency = {}
    for value in values:
        frequency[value] = np.count_nonzero(initial_matrix_gray == value)

    print("Frequency of pixel values:", frequency)

    cdf = {}
    cdf[values[0]] = frequency[values[0]]

    for i in range(1, len(values)):
        cdf[values[i]] = cdf[values[i - 1]] + frequency[values[i]]

    print("Cumulative Distribution Function (CDF):", cdf)
    display_histogram(values, frequency.values(), 'Original Image Histogram')
    return cdf, values, frequency

def display_histogram(values, frequency, title):
    plt.bar(values, frequency, width=1, color='gray', edgecolor='black', label="Pixel frequency")
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    plt.xticks(np.arange(0, 256, 16))
    plt.grid(axis='y')
    plt.show()

def calculate_h(v, cdf, m, n, L=256):
    min_cdf = min(cdf.values())
    return round((L - 1) * (cdf[v] - min_cdf) / ((m * n) - min_cdf))

def equalize_matrix(initial_matrix_gray, cdf, values, frequency):
    h = {}
    row = initial_matrix_gray.shape[0]
    col = initial_matrix_gray.shape[1]
    for value in values:
        h[value] = calculate_h(value, cdf, row, col)

    equalized_matrix_gray = np.zeros_like(initial_matrix_gray)

    for key, value in h.items():
        equalized_matrix_gray[initial_matrix_gray == key] = value

    print("Equalized Cumulative Distribution Function (CDF):", h)
    display_histogram(h.values(), frequency.values(), 'Equalized Image Histogram')
    plt.imshow(equalized_matrix_gray, cmap="gray")
    plt.show()

    return equalized_matrix_gray

def histogram_equalization(path, new_file_name):
    initial_matrix_rgb = load_image(path)
    initial_matrix_gray = convert_to_grayscale(initial_matrix_rgb)
    cdf, values, frequency = calculate_cdf(initial_matrix_gray)
    equalized_img = equalize_matrix(initial_matrix_gray, cdf, values, frequency)
    cv2.imwrite(new_file_name, equalized_img)
    return equalized_img

initial_image_matrix = np.array([[52, 55, 61, 59, 79, 61, 76, 61],
                                  [62, 59, 55, 104, 94, 85, 59, 71],
                                  [63, 65, 66, 113, 144, 104, 63, 72],
                                  [64, 70, 70, 126, 154, 109, 71, 69],
                                  [67, 73, 68, 106, 122, 88, 68, 68],
                                  [68, 79, 60, 70, 77, 66, 58, 75],
                                  [69, 85, 64, 58, 55, 61, 65, 83],
                                  [70, 87, 69, 68, 65, 73, 78, 90]])

plt.imshow(initial_image_matrix, cmap="gray")
plt.show()
cv2.imwrite("image1.png", initial_image_matrix)

cdf, values, frequency = calculate_cdf(initial_image_matrix)
equalized_img = equalize_matrix(initial_image_matrix, cdf, values, frequency)
cv2.imwrite("equalized_image1.png", equalized_img)

final_img = histogram_equalization('./image2.png', 'equalized_image2.png')
final_img = histogram_equalization('./lena.png', 'equalized_image_lena.png')
final_img = histogram_equalization('./image3.jpg', 'equalized_image3.png')
