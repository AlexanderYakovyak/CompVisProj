from src.image_processing.data_visualization import *
from src.image_processing.image_to_blob import image_to_blob
from src.model.filtering import filter_output_probs


f = open('../resources/coco.names', 'rb')
LABELS = list(n.decode('UTF-8').replace('\n', ' ').strip() for n in f.readlines())


def pred_bbs(model, image, threshold_probability=0.9, iou_threshold=0.5, labels=LABELS):
    blob = image_to_blob(image)
    model.setInput(blob)
    output_layers_names = model.getUnconnectedOutLayersNames()
    outputs = model.forward(output_layers_names)

    bboxes, confidences, class_ids = filter_output_probs(threshold_probability=threshold_probability,
                                                         layer_outputs=outputs)

    indices = cv.dnn.NMSBoxes(bboxes=bboxes, scores=confidences, score_threshold=threshold_probability,
                              nms_threshold=iou_threshold)

    draw_bounding_boxes(input_img=image, boxes=bboxes, class_ids=class_ids, confidences=confidences, labels=labels,
                        indicies=indices)
