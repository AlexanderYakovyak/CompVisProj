import cv2 as cv
import random

import tensorflow.data as tfd
import matplotlib.pyplot as plt


def draw_bounding_boxes(input_img, boxes, class_ids, confidences, labels, indicies):

    image_temp = input_img.copy()

    for index in indicies:
        x, y, w, h = boxes[index]
        confidence = confidences[index]

        color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

        cv.rectangle(image_temp, (x, y), (x + w, y + h), color=color, thickness=2)

        text_label = '{}:{:.2f}'.format(labels[class_ids[index]], confidence)

        font_face = cv.FONT_HERSHEY_DUPLEX
        font_scale = 1.5
        font_thickness = 2

        cv.putText(
            img=image_temp,
            text=text_label,
            org=(x + 5, y - 10),
            fontFace=font_face,
            fontScale=font_scale,
            color=color,
            thickness=font_thickness
        )
    plt.imshow(image_temp)
    plt.axis('off')


def show_images_and_bbs(data: tfd.Dataset, grid: list = [6, 3], FIGSIZE: tuple = (30, 40)):

    BB_COLOR = (0, 255, 0)

    plt.figure(figsize=FIGSIZE)
    n_rows, n_cols = grid
    n_images = n_rows * n_cols

    images, boxes = data[0], data[1]

    for index, (image, box) in enumerate(zip(images, boxes)):

        x1, y1, x2, y2 = map(int, box)

        image = cv.rectangle(
            img=image,
            pt1=(x1, y1),
            pt2=(x2, y2),
            thickness=5,
            color=BB_COLOR
        )

        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image)
        plt.axis('off')

        if (index + 1) >= n_images:
            break

    plt.show()
