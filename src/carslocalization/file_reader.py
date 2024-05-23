import os
import cv2
import scipy.io


def read_images_coordinates_and_mask(directory):
    images = []
    coordinates = []
    masks = []

    for filename in os.listdir(directory):
        if (filename.endswith(".jpg") or filename.endswith(".png")) and "dots" not in filename:

            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)

            if img is not None:

                images.append(img)

                txt_filename = filename[:-4] + ".txt"
                txt_path = os.path.join(directory, txt_filename)

                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as file:
                        coords = []
                        for line in file:
                            x, y = map(int, line.split())
                            coords.append((x, y))
                        coordinates.append(coords)

                mat_filename = filename[:-4] + "mask.mat"
                mat_path = os.path.join(directory, mat_filename)

                if os.path.exists(mat_path):
                    mask_data = scipy.io.loadmat(mat_path)
                    masks.append(mask_data)

                    # In case I want to see what overlaying mask with an image does
                    # apply_mask_on_imagee(img_path, mat_path)

    return images, coordinates
