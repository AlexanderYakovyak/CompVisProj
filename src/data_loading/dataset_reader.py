import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import torch
import tensorflow as tf
from tqdm import tqdm

IMAGE_WIDTH = 676
IMAGE_HEIGHT = 380
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)


train_data_dir = 'resources/Kaggle_Cars_Dataset/training_images/'
test_data_dir = 'resources/Kaggle_Cars_Dataset/testing_images/'
train_csv_dir = 'resources/Kaggle_Cars_Dataset/train_solution_bounding_boxes.csv'


TRAINING_IMAGES = 1001
TESTING_IMAGES = 175


def load_image(file_name: str, root_path: str):

    image_path = os.path.join(root_path, file_name)

    original_image = cv.imread(image_path)
    original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
    resized_image = cv.resize(original_image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    resized_image = tf.image.convert_image_dtype(resized_image, dtype=tf.float32)
    resized_image = tf.clip_by_value(resized_image, clip_value_min=0.0, clip_value_max=1.0)

    return resized_image


def load_test_dataset(root_path=test_data_dir, images_num=TESTING_IMAGES):

    images = np.empty(shape=(images_num, *IMAGE_SIZE), dtype=np.float32)

    index = 0
    for filename in os.listdir(root_path):
        if (filename.endswith(".jpg") or filename.endswith(".png")) and "dots" not in filename:

            image = load_image(file_name=filename, root_path=root_path)

            if image is not None:
                images[index] = image
                index += 1

    return images


def load_train_dataset(metadata, root_path=train_data_dir):

    image_paths = metadata['image']

    images = np.empty(shape=(TRAINING_IMAGES, *IMAGE_SIZE), dtype=np.float32)
    bounding_boxes = np.empty(shape=(len(image_paths), 4), dtype=np.float32)

    index = 0
    for image_path in tqdm(image_paths, desc="Loading"):
        try:
            image = load_image(file_name=image_path, root_path=root_path)

            box = np.array([
                metadata['xmin'][index],
                metadata['ymin'][index],
                metadata['xmax'][index],
                metadata['ymax'][index],
            ], dtype=np.float32)

            images[index] = image
            bounding_boxes[index] = box
            index += 1
        except:
            raise IndexError(f"Invalid image path: {image_path}")

    return images, bounding_boxes
