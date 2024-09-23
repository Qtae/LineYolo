import tensorflow as tf
import numpy as np
import os
import math

INPUT_SIZE = 416
GRID_SIZE = 13
NUM_LINES = 3
NUM_CLASSES = 1

def inference(model, image):
    image = tf.expand_dims(image, axis=0)
    preds = model.predict(image)

    confidence_threshold = 0.5
    line_confidences = preds[0, ..., 0]
    rhos = preds[0, ..., 1]
    thetas = preds[0, ..., 2]

    detected_lines = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            for k in range(NUM_LINES):
                if line_confidences[i, j, k] > confidence_threshold:
                    rho = rhos[i, j, k]
                    theta = thetas[i, j, k]
                    detected_lines.append([rho, theta])

    return detected_lines

test_image_path = 'path/to/test_image.jpg'
image = tf.io.read_file(test_image_path)
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [INPUT_SIZE, INPUT_SIZE])
image = image / 255.0  # 정규화

#detected_lines = inference(model, image)
#
# print("검출된 직선 수:", len(detected_lines))
# for line in detected_lines:
#     print("직선 파라미터 (rho, theta):", line)