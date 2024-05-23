import numpy as np


def compute_mean_std(images):
    mean = np.array([0., 0., 0.])
    std_temp = np.array([0., 0., 0.])
    std = np.array([0., 0., 0.])

    num_samples = len(images)

    for image in images:
        image = image.astype(float) / 255.

        for j in range(3):
            mean[j] += np.mean(image[:, :, ])

    mean = (mean/num_samples)

    for image in images:
        image = image.astype(float) / 255.
        for j in range(3):
            std_temp[j] += ((image[:, :, j] - mean[j])**2).sum()/(image.shape[0]*image.shape[1])

    std = np.sqrt(std_temp/num_samples)

    return mean, std
