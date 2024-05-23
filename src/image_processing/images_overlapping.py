import cv2


def overlay_coordinates_and_mask(images, coordinates):
    overlay_images = []

    for img, coords in zip(images, coordinates):
        overlay_img = img.copy()

        for coord in coords:
            x, y = coord
            cv2.circle(overlay_img, (x, y), 3, (0, 255, 0), -1)

        overlay_images.append(overlay_img)

    return overlay_images
