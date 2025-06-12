# FILE: src/model_mobilenet/handler.py (Đã có Cải tiến 1)

import cv2
import numpy as np

def process_frame_with_mobilenet(net, image_bgr, threshold=0.1):
    annotated_image = image_bgr.copy()
    image_height, image_width, _ = annotated_image.shape
    
    aspect_ratio = image_width / image_height
    in_height = 368
    in_width = int(((aspect_ratio * in_height) * 8) // 8)
    
    inpBlob = cv2.dnn.blobFromImage(
        annotated_image, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=False, crop=False
    )
    
    net.setInput(inpBlob)
    output = net.forward()

    points = []
    for i in range(22):
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (image_width, image_height))
        _, prob, _, point = cv2.minMaxLoc(probMap)
        
        # SỬ DỤNG THAM SỐ THRESHOLD
        if prob > threshold:
            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)
    
    POSE_PAIRS = [ 
        [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],
        [0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] 
    ]
    for pair in POSE_PAIRS:
        partA, partB = pair[0], pair[1]
        if points[partA] and points[partB]:
            cv2.line(annotated_image, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(annotated_image, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(annotated_image, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            
    return annotated_image