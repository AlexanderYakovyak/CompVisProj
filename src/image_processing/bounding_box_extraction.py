import cv2
import numpy as np


min_width = 5
min_height = 5

offset = 6

counting_line_position = 400


detections = []
cars = 0


def get_center(x, y, w, h):
    x_center = int(w / 2)
    y_center = int(h / 2)
    cx = x + x_center
    cy = y + y_center
    return cx, cy


background_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG()


def bounding_box(image):

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (3, 3), 5)
    foreground_mask = background_subtractor.apply(blurred_img)
    dilated_mask = cv2.dilate(foreground_mask, np.ones((5, 5), np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel)
    closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(closed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(image, (25, counting_line_position), (1200, counting_line_position), (255, 127, 0), 3)

    for i, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        is_valid_contour = (w >= min_width) and (h >= min_height)

        if not is_valid_contour:
            continue

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        center = get_center(x, y, w, h)
        print(center)
        detections.append(center)

        cv2.circle(image, center, 4, (0, 0, 255), -1)

    cv2.imshow('Frame', image)
    cv2.waitKey(0)

