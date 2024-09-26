import tensorflow as tf
import numpy as np
import os
import cv2
import datetime

# Constants
INPUT_SIZE = 416
GRID_SIZE = 13
NUM_LINES = 3
NUM_CLASSES = 1
TRAIN_VAL_SPLIT = 0.1

def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]

def convolutional(input_layer, filters_shape, strides=1, padding='same',
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01), downsample=False, activate=True,
                  bn=True, activate_type='leaky'):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides,
                                  padding=padding, use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=kernel_initializer,
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn:
        conv = tf.keras.layers.BatchNormalization()(conv)

    if activate:
        if activate_type == "leaky":
            conv = tf.keras.layers.LeakyReLU(alpha=0.1)(conv)
        elif activate_type == "relu":
            conv = tf.keras.layers.ReLU()(conv)

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
    input_data = convolutional(input_data, (1, 1, 64, 64))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(input_data, (3, 3, 64, 128))
    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(input_data, (3, 3, 64, 64))
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 64, 64))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(input_data, (1, 1, 128, 128))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(input_data, (3, 3, 128, 256))
    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(input_data, (3, 3, 128, 128))
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 128, 128))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(input_data, (1, 1, 256, 256))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(input_data, (3, 3, 256, 512))

    return input_data

def create_line_detection_model():
    inputs = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])

    x = cspdarknet53_tiny(inputs)

    global GRID_SIZE
    GRID_SIZE = x.shape[1]

    # NUM_LINES=3으로 설정됨에 따라, 출력 채널 수도 변경
    x = tf.keras.layers.Conv2D(NUM_LINES * (1 + 2), 1, padding='same')(x)
    x = tf.keras.layers.Reshape((GRID_SIZE, GRID_SIZE, NUM_LINES, 3))(x)

    line_confidence = tf.sigmoid(x[..., 0])
    rho = tf.sigmoid(x[..., 1])
    theta = tf.sigmoid(x[..., 2])

    outputs = tf.concat([
        line_confidence[..., tf.newaxis],
        rho[..., tf.newaxis],
        theta[..., tf.newaxis],
    ], axis=-1)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)

def line_detection_loss(y_true, y_pred):
    obj_mask = y_true[..., 0]
    pred_confidence = y_pred[..., 0]

    confidence_loss = tf.keras.losses.binary_crossentropy(obj_mask, pred_confidence)
    confidence_loss = tf.reduce_mean(confidence_loss)

    rho_loss = tf.square(y_true[..., 1] - y_pred[..., 1])
    rho_loss = obj_mask * rho_loss
    rho_loss = tf.reduce_sum(rho_loss) / (tf.reduce_sum(obj_mask) + 1e-6)

    theta_true = y_true[..., 2]
    theta_pred = y_pred[..., 2]

    theta_loss = tf.square(theta_true - theta_pred)
    theta_loss = obj_mask * theta_loss
    theta_loss = tf.reduce_sum(theta_loss) / (tf.reduce_sum(obj_mask) + 1e-6)

    total_loss = 5 * confidence_loss + rho_loss + theta_loss
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
        def _parse_function(image_path, label_path):
            # Decode bytes to strings
            image_path = image_path.numpy().decode()
            label_path = label_path.numpy().decode()

            # Read and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Image not found or unable to read: {image_path}")
            original_height, original_width = image.shape[:2]
            image_resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
            image_normalized = image_resized / 255.0

            # Read and process labels
            with open(label_path, 'r') as f:
                lines = f.readlines()

            lines = [line.strip() for line in lines if line.strip()]
            labels = [list(map(float, line.split())) for line in lines]

            label_tensor = np.zeros((GRID_SIZE, GRID_SIZE, NUM_LINES, 3), dtype=np.float32)

            # Dictionary to keep track of the number of lines assigned to each grid cell
            grid_counts = {}

            for label in labels:
                if len(label) < 2:
                    continue  # Skip invalid labels
                rho_norm, theta = label[:2]
                theta_norm = theta / np.pi

                # Compute (x, y) coordinates of the closest point to origin
                rho = rho_norm * np.sqrt(original_width ** 2 + original_height ** 2)

                x = rho * np.cos(theta)
                y = rho * np.sin(theta)

                # Normalize coordinates to [0, 1]
                x_norm = (x + (original_width / 2)) / original_width
                y_norm = (y + (original_height / 2)) / original_height

                # Map normalized coordinates to grid indices
                grid_x = int(x_norm * GRID_SIZE)
                grid_y = int(y_norm * GRID_SIZE)

                # Clip to ensure indices are within [0, GRID_SIZE-1]
                grid_x = np.clip(grid_x, 0, GRID_SIZE - 1)
                grid_y = np.clip(grid_y, 0, GRID_SIZE - 1)

                key = (grid_y, grid_x)

                if key not in grid_counts:
                    grid_counts[key] = 0

                if grid_counts[key] < NUM_LINES:
                    k = grid_counts[key]
                    grid_counts[key] += 1

                    label_tensor[grid_y, grid_x, k, 0] = 1.0  # confidence
                    label_tensor[grid_y, grid_x, k, 1] = rho_norm  # rho_norm
                    label_tensor[grid_y, grid_x, k, 2] = theta_norm

            return image_normalized.astype(np.float32), label_tensor.astype(np.float32)

        # Use tf.py_function to wrap the Python function
        image, label_tensor = tf.py_function(_parse_function, [image_path, label_path], [tf.float32, tf.float32])
        image.set_shape([INPUT_SIZE, INPUT_SIZE, 3])
        label_tensor.set_shape([GRID_SIZE, GRID_SIZE, NUM_LINES, 3])

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

def inference_and_display(model, image_paths):
    for imgidx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Unable to read image: {image_path}")
            continue
        original_height, original_width = image.shape[:2]

        image_resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        input_image = image_resized / 255.0
        input_tensor = tf.expand_dims(input_image, axis=0)  # [1, INPUT_SIZE, INPUT_SIZE, 3]

        preds = model.predict(input_tensor)
        preds = preds[0]  # [GRID_SIZE, GRID_SIZE, NUM_LINES, 5]

        # Calculate scaling factors
        original_diagonal = np.sqrt(original_width ** 2 + original_height ** 2)

        detected_lines = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                for k in range(NUM_LINES):
                    confidence = preds[i, j, k, 0]
                    if confidence > 0.2:
                        rho_norm = preds[i, j, k, 1]
                        theta_norm = preds[i, j, k, 2]
                        theta = theta_norm * np.pi
                        rho = rho_norm * original_diagonal
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

        print(f"Detected Lines in {image_path}: {detected_lines}")
        cv2.imshow('Detected Lines', image_with_lines)
        cv2.imwrite(f'./{imgidx}.png', image_with_lines)
        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()

def run_inference_on_folder(model, test_dir):
    test_image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    test_image_files.sort()
    test_image_paths = [os.path.join(test_dir, f) for f in test_image_files]
    inference_and_display(model, test_image_paths)


if __name__ == '__main__':
    data_dir = r'D:\Work\04_Samsung_Pyeongtak_WIND2\Data'
    batch_size = 16
    valid_ratio = 0.2  # Validation ratio

    # Load data
    train_dataset, valid_dataset = load_data(data_dir, batch_size, valid_ratio)
    steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    validation_steps = tf.data.experimental.cardinality(valid_dataset).numpy()

    # Model creation
    model = create_line_detection_model()
    model.summary()

    # Model compilation
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss=line_detection_loss)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True)

    # Training
    epochs = 200

    model.fit(train_dataset,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid_dataset,
              validation_steps=validation_steps,
              epochs=epochs,
              callbacks=[early_stopping])

    # Save the model
    save_dir = os.path.join('./models', datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save(save_dir, overwrite=True)

    # Load the model for inference
    # model = tf.keras.models.load_model(r'D:\Work\04_Samsung_Pyeongtak_WIND2\LineYoloTest\models\2024-09-25_16h37m17s_sota',
    #                                    compile=False)

    # Run inference
    test_dir = r'D:\Work\04_Samsung_Pyeongtak_WIND2\Data_orig'
    run_inference_on_folder(model, test_dir)
