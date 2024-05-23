import numpy as np

IMAGE_WIDTH = 676
IMAGE_HEIGHT = 380
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)


def filter_output_probs(threshold_probability, layer_outputs):

    filtered_boxes = []
    filtered_confidences = []
    filtered_class_ids = []

    for output in layer_outputs:
        for detection in output:

            probabilities = detection[5:]
            class_id = np.argmax(probabilities)

            confidence = probabilities[class_id]

            if confidence > threshold_probability:

                box = detection[0:4] * np.array([IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT])

                center_x, center_y, box_width, box_height = box.astype('int')
                x_min = int(center_x - (box_width / 2))
                y_min = int(center_y - (box_height / 2))

                filtered_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                filtered_confidences.append(float(confidence))
                filtered_class_ids.append(class_id)

    return filtered_boxes, filtered_confidences, filtered_class_ids
