import cv2 as cv


def image_to_blob(image):

    blob = cv.dnn.blobFromImage(
        image=image,
        size=(416, 416),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )

    return blob
