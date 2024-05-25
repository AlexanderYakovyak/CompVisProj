from data_loading.dataset_reader import *
from image_processing.data_visualization import *
from image_processing.image_to_blob import image_to_blob
from model.filtering import filter_output_probs

train_csv_dir = '../resources/Kaggle_Cars_Dataset/train_solution_bounding_boxes.csv'
test_trancos_dir = "../../TRANCOS_v3/images_to_test/"

f = open('../resources/coco.names', 'rb')
labels = list(n.decode('UTF-8').replace('\n', ' ').strip() for n in f.readlines())

random_test_images = '../resources/random_images/'

def pred_bbs(image, threshold_probability=0.9, iou_threshold=0.5, labels=labels):
    blob = image_to_blob(image)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers_names)

    bboxes, confidences, class_ids = filter_output_probs(threshold_probability=threshold_probability,
                                                         layer_outputs=outputs)

    indices = cv.dnn.NMSBoxes(bboxes=bboxes, scores=confidences, score_threshold=threshold_probability,
                              nms_threshold=iou_threshold)

    draw_bounding_boxes(input_img=image, boxes=bboxes, class_ids=class_ids, confidences=confidences, labels=labels,
                        indicies=indices)


def test_random_images(images_to_test, n_images=1):
    plt.figure()

    for i in range(n_images):
        plt.subplot(n_images, 1, i + 1)
        image = images_to_test[3]
        #image = images_to_test[np.random.randint(len(images_to_test))]
        pred_bbs(image)

    plt.show()


if __name__ == '__main__':
    #metadata = pd.read_csv(train_csv_dir)

    images = load_test_dataset(root_path=random_test_images, images_num=6)

    #show_images_and_bbs(data=dataset[:5],GRID=[5, 1])

    net = cv.dnn.readNet('../resources/yolo-data/yolov3.weights', '../resources/yolo-data/yolov3.cfg')

    test_random_images(images)
