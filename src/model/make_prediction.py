from image_processing.image_to_blob import image_to_blob


def pred_bbs(image, threshold_probability = 0.9, iou_threshold = 0.5, labels = labels):

    blob = image_to_blob(image)

    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers_names)

    # Filter the inputs based on probability
    bboxes, confidences, class_ids = filter_output_probs(threshold_probability=threshold_probability,
                                                         layer_outputs=outputs)

    # Apply Non Max Suppression
    indicies = cv.dnn.NMSBoxes(bboxes=bboxes, scores=confidences, score_threshold=threshold_probability,
                               nms_threshold=iou_threshold)

    # Draw the Bounding Box
    draw_bounding_boxes(input_img=image, boxes=bboxes, class_ids=class_ids, confidences=confidences, labels=labels,
                        indicies=indicies)