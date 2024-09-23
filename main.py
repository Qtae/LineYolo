import tensorflow as tf
import numpy as np
import os
import cv2

# Constants
INPUT_SIZE = 416
GRID_SIZE = 13
NUM_LINES = 3
NUM_CLASSES = 1
TRAIN_VAL_SPLIT = 0.2


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]


def convolutional(input_layer, filters_shape, strides=1, padding='same',
                  kernal_initializer=tf.random_normal_initializer(stddev=0.01), downsample=False, activate=True,
                  bn=True, activate_type='leaky'):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides,
                                  padding=padding, use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=kernal_initializer,
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn: conv = BatchNormalization()(conv)

    if activate == True:
        if activate_type == "leaky":
            conv = tf.keras.activations.relu(conv, alpha=0.1)
    return conv


def cspdarknet53_tiny(input_data):
    input_data = convolutional(input_data, (3, 3, 3, 32), downsample=True)
    input_data = convolutional(input_data, (3, 3, 32, 64), downsample=True)
    input_data = convolutional(input_data, (3, 3, 64, 64))

    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(input_data, (3, 3, 32, 32))
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 32, 32))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(input_data, (1, 1, 32, 64))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(input_data, (3, 3, 64, 128))
    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(input_data, (3, 3, 64, 64))
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 64, 64))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(input_data, (1, 1, 64, 128))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(input_data, (3, 3, 128, 256))
    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(input_data, (3, 3, 128, 128))
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 128, 128))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(input_data, (1, 1, 128, 256))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(input_data, (3, 3, 512, 512))

    return input_data


def create_line_detection_model():
    inputs = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])

    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)    # [416,416,32]
    x = tf.keras.layers.MaxPooling2D(2)(x)  # [208,208,32]
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)          # [208,208,64]
    x = tf.keras.layers.MaxPooling2D(2)(x)  # [104,104,64]
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)         # [104,104,128]
    x = tf.keras.layers.MaxPooling2D(2)(x)  # [52,52,128]
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)         # [52,52,256]

    x = tf.keras.layers.MaxPooling2D(2)(x)  # [26,26,256]
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)         # [26,26,512]
    x = tf.keras.layers.MaxPooling2D(2)(x)  # [13,13,512]

    x = tf.keras.layers.Conv2D(NUM_LINES * (1 + 2 + NUM_CLASSES), 1, padding='same')(x)  # [13,13,12]
    x = tf.keras.layers.Reshape((GRID_SIZE, GRID_SIZE, NUM_LINES, 3 + NUM_CLASSES))(x)    # [13,13,3,4]

    line_confidence = tf.sigmoid(x[..., 0])
    rho = tf.sigmoid(x[..., 1])
    theta = x[..., 2]

    theta = tf.tanh(theta) * np.pi

    if NUM_CLASSES > 1:
        class_probs = tf.nn.softmax(x[..., 3:])
    else:
        class_probs = tf.sigmoid(x[..., 3])

    outputs = tf.concat([line_confidence[..., tf.newaxis],
                         rho[..., tf.newaxis],
                         theta[..., tf.newaxis],
                         class_probs[..., tf.newaxis]],
                        axis=-1)
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


# def create_line_detection_model():
#     inputs = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
#
#     x = cspdarknet53_tiny(inputs)
#
#     global GRID_SIZE
#     GRID_SIZE = x.shape[1]
#
#     x = tf.keras.layers.Conv2D(NUM_LINES * (1 + 2 + NUM_CLASSES), 1, padding='same')(x)
#     x = tf.keras.layers.Reshape((GRID_SIZE, GRID_SIZE, NUM_LINES, 3 + NUM_CLASSES))(x)
#
#     line_confidence = tf.sigmoid(x[..., 0])
#     rho = tf.sigmoid(x[..., 1])
#     theta = x[..., 2]
#
#     theta = tf.tanh(theta) * np.pi
#
#     if NUM_CLASSES > 1:
#         class_probs = tf.nn.softmax(x[..., 3:])
#     else:
#         class_probs = tf.sigmoid(x[..., 3])
#
#     outputs = tf.concat([line_confidence[..., tf.newaxis],
#                          rho[..., tf.newaxis],
#                          theta[..., tf.newaxis],
#                          class_probs[..., tf.newaxis]],
#                         axis=-1)
#     return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def line_detection_loss(y_true, y_pred):
    obj_mask = y_true[..., 0]
    pred_confidence = y_pred[..., 0]
    lambda_obj = 5.0
    lambda_noobj = 1.0

    confidence_loss_obj = lambda_obj * obj_mask * tf.keras.backend.binary_crossentropy(obj_mask, pred_confidence)
    confidence_loss_noobj = lambda_noobj * (1 - obj_mask) * tf.keras.backend.binary_crossentropy(obj_mask, pred_confidence)
    confidence_loss = tf.reduce_sum(confidence_loss_obj + confidence_loss_noobj) / (GRID_SIZE * GRID_SIZE * NUM_LINES)

    rho_loss = tf.square(y_true[..., 1] - y_pred[..., 1])
    theta_diff = y_true[..., 2] - y_pred[..., 2]
    theta_diff = tf.math.floormod(theta_diff + np.pi, 2 * np.pi) - np.pi
    theta_loss = tf.square(theta_diff / np.pi)
    param_loss = rho_loss + theta_loss
    param_loss = obj_mask * param_loss
    param_loss = tf.reduce_sum(param_loss) / (tf.reduce_sum(obj_mask) + 1e-6)

    lambda_coord = 5.0
    total_loss = confidence_loss + lambda_coord * param_loss
    return total_loss


def load_data(data_dir, batch_size, valid_ratio):
    # Get all image and label file paths
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort()

    image_paths = [os.path.join(data_dir, f) for f in image_files]
    label_paths = [os.path.join(data_dir, os.path.splitext(f)[0] + '.txt') for f in image_files]

    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))

    def parse_function(image_path, label_path):
        image_raw = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image_raw, channels=3)
        original_height = tf.shape(image)[0]
        original_width = tf.shape(image)[1]

        # Resize and normalize the image
        image = tf.image.resize(image, [INPUT_SIZE, INPUT_SIZE])
        image = image / 255.0

        # Read and process labels
        labels = tf.io.read_file(label_path)
        labels = tf.strings.strip(labels)
        lines = tf.strings.split(labels, '\n')

        splits = tf.strings.split(lines, ' ')
        labels = tf.strings.to_number(splits, out_type=tf.float32)
        labels = labels.to_tensor(default_value=0.0)  # [num_labels, 2]

        num_labels = tf.shape(labels)[0]

        # Process the image and labels based on the number of labels
        def process_with_labels():
            rho = labels[:, 0]
            theta = labels[:, 1]
            image_diagonal = tf.sqrt(tf.cast(original_width ** 2 + original_height ** 2, tf.float32))
            rho = rho / image_diagonal
            cos_theta = tf.cos(theta)
            sin_theta = tf.sin(theta)
            grid_x = tf.linspace(0.0, tf.cast(original_width, tf.float32), GRID_SIZE + 1)
            grid_y = tf.linspace(0.0, tf.cast(original_height, tf.float32), GRID_SIZE + 1)
            grid_x = (grid_x[:-1] + grid_x[1:]) / 2
            grid_y = (grid_y[:-1] + grid_y[1:]) / 2
            grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
            grid_centers = tf.stack([grid_x, grid_y], axis=-1)
            x = tf.expand_dims(grid_centers[..., 0], axis=0)
            y = tf.expand_dims(grid_centers[..., 1], axis=0)
            cos_theta_expanded = tf.reshape(cos_theta, [-1, 1, 1])
            sin_theta_expanded = tf.reshape(sin_theta, [-1, 1, 1])
            rho_expanded = tf.reshape(rho, [-1, 1, 1])
            distances = x * cos_theta_expanded + y * sin_theta_expanded - rho_expanded
            distances = tf.abs(distances)
            threshold = 20.0
            mask = distances < threshold
            mask_transposed = tf.transpose(mask, perm=[1, 2, 0])

            def select_labels_per_cell(cell_mask):
                label_indices = tf.where(cell_mask)[:, 0]
                label_indices = tf.cast(label_indices, tf.int32)
                num_labels_in_cell = tf.shape(label_indices)[0]
                num_to_select = tf.minimum(num_labels_in_cell, NUM_LINES)
                selected_indices = label_indices[:num_to_select]
                padding_size = NUM_LINES - num_to_select
                selected_indices = tf.pad(selected_indices, [[0, padding_size]], constant_values=-1)
                return selected_indices

            cell_labels = tf.map_fn(
                select_labels_per_cell,
                elems=tf.reshape(mask_transposed, [-1, num_labels]),
                fn_output_signature=tf.int32)

            cell_labels = tf.reshape(cell_labels, [GRID_SIZE, GRID_SIZE, NUM_LINES])

            labels_padded = tf.concat([labels, tf.zeros([1, 2], dtype=tf.float32)], axis=0)

            cell_labels = tf.where(cell_labels >= 0, cell_labels, num_labels)

            rho_theta = tf.gather(labels_padded, cell_labels)

            exists = tf.cast(cell_labels != num_labels, tf.float32)

            label_tensor = tf.concat([tf.expand_dims(exists, axis=-1),
                                      rho_theta,
                                      tf.zeros((GRID_SIZE, GRID_SIZE, NUM_LINES, NUM_CLASSES), dtype=tf.float32)],
                                     axis=-1)

            return image, label_tensor

        def process_without_labels():
            label_tensor = tf.zeros((GRID_SIZE, GRID_SIZE, NUM_LINES, 3 + NUM_CLASSES), dtype=tf.float32)
            return image, label_tensor

        image, label_tensor = tf.cond(num_labels > 0, process_with_labels, process_without_labels)
        return image, label_tensor

    # Shuffle and map the dataset
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)

    # Calculate the number of validation samples
    num_samples = len(image_paths)
    num_valid_samples = int(num_samples * valid_ratio)

    # Split the dataset
    valid_dataset = dataset.take(num_valid_samples)
    train_dataset = dataset.skip(num_valid_samples)

    # Batch and prefetch datasets
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, valid_dataset

# Configuration
data_dir = r'D:\Work\04_Samsung_Pyeongtak_WIND2\Data'
batch_size = 16
valid_ratio = 0.2  # Validation ratio

# Load data
train_dataset, valid_dataset = load_data(data_dir, batch_size, valid_ratio)
steps_per_epoch = sum(1 for _ in train_dataset)
validation_steps = sum(1 for _ in valid_dataset)

# Model creation
model = create_line_detection_model()
model.summary()

# Model compilation
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=line_detection_loss)

# Training
epochs = 100
model.fit(train_dataset,
          steps_per_epoch=steps_per_epoch,
          validation_data=valid_dataset,
          validation_steps=validation_steps,
          epochs=epochs)


def inference_and_display(model, image_paths):
    for image_path in image_paths:
        image = cv2.imread(image_path)
        original_height, original_width = image.shape[:2]

        image_resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        input_image = image_resized / 255.0
        input_tensor = tf.expand_dims(input_image, axis=0)  # [1, INPUT_SIZE, INPUT_SIZE, 3]

        preds = model.predict(input_tensor)
        preds = preds[0]

        scale_x = original_width / INPUT_SIZE
        scale_y = original_height / INPUT_SIZE

        resized_diagonal = np.sqrt(INPUT_SIZE ** 2 + INPUT_SIZE ** 2)
        original_diagonal = np.sqrt(original_width ** 2 + original_height ** 2)

        rho_scale = original_diagonal / resized_diagonal

        detected_lines = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                for k in range(NUM_LINES):
                    confidence = preds[i, j, k, 0]
                    if confidence > 0.5:
                        rho = preds[i, j, k, 1]
                        theta = preds[i, j, k, 2]

                        rho = rho * resized_diagonal
                        rho = rho * rho_scale

                        detected_lines.append((rho, theta))

        image_with_lines = image.copy()

        for rho, theta in detected_lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)


        print(detected_lines)
        cv2.imshow('Detected Lines', image_with_lines)
        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()


def run_inference_on_folder(model, test_dir):
    test_image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    test_image_files.sort()
    test_image_paths = [os.path.join(test_dir, f) for f in test_image_files]
    inference_and_display(model, test_image_paths)

test_dir = r'D:\Work\04_Samsung_Pyeongtak_WIND2\Data_orig'
run_inference_on_folder(model, test_dir)