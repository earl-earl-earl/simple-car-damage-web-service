def postprocess_detections(outputs, img_width, img_height, confidence_thres=0.3, iou_thres=0.3):
    outputs = np.transpose(np.squeeze(outputs[0]))
    rows = outputs.shape[0]
    boxes, scores, class_ids = [], [], []

    for i in range(rows):
        class_scores = outputs[i][4:]
        max_score = np.amax(class_scores)
        if max_score >= confidence_thres:
            class_id = np.argmax(class_scores)
            x, y, w, h = outputs[i][:4]
            left, top, width, height = int(x - w / 2), int(y - h / 2), int(w), int(h)
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, left + width, top + height])

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)

    detected_labels = []
    if indices is not None:  # Check if any boxes survived NMS
        try:
            if isinstance(indices, tuple):
                # Convert tuple to a NumPy array (handling the case where it might be empty)
                if indices: # Check if the tuple is not empty
                    indices = np.array(indices)
                else:
                    indices = np.array([]) #empty array
            elif not isinstance(indices, np.ndarray):
                # If it's not a tuple or ndarray, try to convert it
                indices = np.array(indices)

            if indices.size > 0:  # Check if the array is not empty.  Prevents error if empty list or array
                indices = indices.flatten() # Flatten regardless

                for i in indices:
                    detected_labels.append(damage_classes.get(str(int(i))), "Unknown")

        except Exception as e:
            print(f"Error processing indices: {e}")
            return ["Error Processing Detections"]

    return detected_labels if detected_labels else ["No Damage"]
