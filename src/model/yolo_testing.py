from src.data_loading.dataset_reader import *
from src.model.predict_bb import pred_bbs

random_test_images = '../resources/random_images/'


def test_random_images_yolo(model, images_to_test, n_images=1):
    plt.figure()
    for i in range(n_images):
        plt.subplot(n_images, 1, i + 1)
        image = images_to_test[1]
        #image = images_to_test[np.random.randint(len(images_to_test))]
        pred_bbs(model, image)
    plt.show()


def test_yolo(n_images=6):
    images = load_test_dataset(root_path=random_test_images, images_num=n_images)
    net = cv.dnn.readNet('../resources/yolo-data/yolov3.weights', '../resources/yolo-data/yolov3.cfg')
    test_random_images_yolo(net,images)