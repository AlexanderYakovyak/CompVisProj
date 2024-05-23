import os


def read_images_coordinates_filenames(directory):
    images_names = []
    coordinates_names = []

    for filename in os.listdir(directory):
        if (filename.endswith(".jpg") or filename.endswith(".png")) and "dots" not in filename:
            images_names.append(filename)

            if filename is not None:
                txt_filename = filename[:-4] + ".txt"
                coordinates_names.append(txt_filename)

    return images_names, coordinates_names
