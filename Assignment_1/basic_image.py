import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    initial_matrix_rgb = np.array(img)

    plt.imshow(initial_matrix_rgb)
    plt.axis('off')
    plt.show()

    dimensions = img.shape
    print(dimensions)
    print('Height of Image:', dimensions[0])
    print('Width of Image:', dimensions[1])
    print('Number of Dimensions:', dimensions[2])

    row = img.shape[0]
    col = img.shape[1]

    red_layer = np.zeros_like(img)
    for i in range(row):
        for j in range(col):
            red_layer[i, j, 0] = img[i, j, 2]  # Red channel

    green_layer = np.zeros_like(img)
    for i in range(row):
        for j in range(col):
            green_layer[i, j, 1] = img[i, j, 1]  # Green channel

    blue_layer = np.zeros_like(img)
    for i in range(row):
        for j in range(col):
            blue_layer[i, j, 2] = img[i, j, 0]  # Blue channel

    grayscale = np.zeros((row, col), dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            grayscale[i, j] = round((img[i, j, 0] + img[i, j, 1] + img[i, j, 2]) / 3)

    grayscale2 = np.zeros((row, col), dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            grayscale2[i, j] = round(
                0.299 * img[i, j, 2] +  # Red channel
                0.587 * img[i, j, 1] +  # Green channel
                0.114 * img[i, j, 0]    # Blue channel
            )

    print("Grayscale Image using (R+G+B)/3")
    plt.imshow(grayscale, cmap='gray')
    plt.axis('off')
    plt.show()

    print("Grayscale Image using adjusted formula")
    plt.imshow(grayscale2, cmap='gray')
    plt.axis('off')
    plt.show()

    print("Image R layer")
    plt.imshow(red_layer)
    plt.axis('off')
    plt.show()

    print("Image G layer")
    plt.imshow(green_layer)
    plt.axis('off')
    plt.show()

    print("Image B layer")
    plt.imshow(blue_layer)
    plt.axis('off')
    plt.show()

load_image('image2.png')
load_image('image3.jpg')
load_image('lena.png')
