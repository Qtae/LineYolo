import tensorflow as tf
import numpy as np
import os
import math

INPUT_SIZE = 416
GRID_SIZE = 13
NUM_LINES = 3
NUM_CLASSES = 1


def create_line_detection_model():
    inputs = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])

    # 기존 레이어들
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)    # [416,416,32]
    x = tf.keras.layers.MaxPooling2D(2)(x)  # [208,208,32]
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)          # [208,208,64]
    x = tf.keras.layers.MaxPooling2D(2)(x)  # [104,104,64]
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)         # [104,104,128]
    x = tf.keras.layers.MaxPooling2D(2)(x)  # [52,52,128]
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)         # [52,52,256]

    # 추가된 다운샘플링 레이어들
    x = tf.keras.layers.MaxPooling2D(2)(x)  # [26,26,256]
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)         # [26,26,512]
    x = tf.keras.layers.MaxPooling2D(2)(x)  # [13,13,512]

    # 마지막 출력 레이어
    x = tf.keras.layers.Conv2D(NUM_LINES * (1 + 2 + NUM_CLASSES), 1, padding='same')(x)  # [13,13,12]
    x = tf.keras.layers.Reshape((GRID_SIZE, GRID_SIZE, NUM_LINES, 3 + NUM_CLASSES))(x)    # [13,13,3,4]

    # 이후 동일
    line_confidence = tf.sigmoid(x[..., 0])  # 직선 존재 확률
    rho = x[..., 1]  # rho: [0, 1]로 정규화된 값
    theta = x[..., 2]  # theta: [-π, π] 범위

    # theta를 [-π, π] 범위로 만들기 위한 활성화 함수
    theta = tf.tanh(theta) * np.pi

    if NUM_CLASSES > 1:
        class_probs = tf.nn.softmax(x[..., 3:])
    else:
        class_probs = tf.sigmoid(x[..., 3])

    outputs = tf.concat([line_confidence[..., tf.newaxis], rho[..., tf.newaxis], theta[..., tf.newaxis], class_probs[..., tf.newaxis]], axis=-1)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def line_detection_loss(y_true, y_pred):
    """
    y_true와 y_pred의 형태:
    [batch_size, GRID_SIZE, GRID_SIZE, NUM_LINES, 3 + NUM_CLASSES]
    """
    # 직선 존재 여부 손실 (Binary Cross-Entropy)
    obj_mask = y_true[..., 0]  # [batch_size, GRID_SIZE, GRID_SIZE, NUM_LINES]
    pred_confidence = y_pred[..., 0]  # [batch_size, GRID_SIZE, GRID_SIZE, NUM_LINES]

    # 요소별로 binary cross-entropy 손실 계산
    confidence_loss = tf.keras.backend.binary_crossentropy(obj_mask,
                                                           pred_confidence)  # [batch_size, GRID_SIZE, GRID_SIZE, NUM_LINES]

    # 파라미터 손실
    # rho 손실 (Mean Squared Error)
    rho_loss = tf.square(y_true[..., 1] - y_pred[..., 1])  # [batch_size, GRID_SIZE, GRID_SIZE, NUM_LINES]

    # theta 손실 (주기성 고려)
    theta_diff = y_true[..., 2] - y_pred[..., 2]
    theta_diff = tf.math.floormod(theta_diff + np.pi, 2 * np.pi) - np.pi  # [-π, π] 범위로 조정
    theta_loss = tf.square(theta_diff)  # [batch_size, GRID_SIZE, GRID_SIZE, NUM_LINES]

    param_loss = rho_loss + theta_loss  # [batch_size, GRID_SIZE, GRID_SIZE, NUM_LINES]
    param_loss = obj_mask * param_loss  # 직선이 존재하는 위치에 대해서만 계산

    # 총 손실 계산
    total_loss = confidence_loss + param_loss  # [batch_size, GRID_SIZE, GRID_SIZE, NUM_LINES]
    total_loss = tf.reduce_mean(total_loss)  # 배치 전체에 대해 평균 계산

    return total_loss


def load_data(data_dir, batch_size):
    # 이미지 파일과 레이블 파일의 전체 경로를 생성
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort()

    image_paths = [os.path.join(data_dir, f) for f in image_files]
    label_paths = [os.path.join(data_dir, os.path.splitext(f)[0] + '.txt') for f in image_files]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))

    def parse_function(image_path, label_path):
        # 이미지 로드 및 전처리
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [INPUT_SIZE, INPUT_SIZE])
        image = image / 255.0  # 정규화

        # 레이블 파일 로드
        labels = tf.io.read_file(label_path)
        labels = tf.strings.strip(labels)
        lines = tf.strings.split(labels, '\n')

        # 각 줄을 공백으로 분할하여 (rho, theta)로 변환
        splits = tf.strings.split(lines, ' ')
        labels = tf.strings.to_number(splits, out_type=tf.float32)
        labels = labels.to_tensor(default_value=0.0)  # [num_labels, 2]

        num_labels = tf.shape(labels)[0]

        # 레이블에서 rho와 theta 추출
        rho = labels[:, 0]  # [num_labels]
        theta = labels[:, 1]  # [num_labels]

        cos_theta = tf.cos(theta)  # [num_labels]
        sin_theta = tf.sin(theta)  # [num_labels]

        # 그리드 셀의 좌표 계산
        grid_x = tf.linspace(0.0, 1.0, GRID_SIZE + 1)
        grid_y = tf.linspace(0.0, 1.0, GRID_SIZE + 1)
        grid_x = (grid_x[:-1] + grid_x[1:]) / 2  # [GRID_SIZE]
        grid_y = (grid_y[:-1] + grid_y[1:]) / 2  # [GRID_SIZE]
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)  # [GRID_SIZE, GRID_SIZE]
        grid_centers = tf.stack([grid_x, grid_y], axis=-1)  # [GRID_SIZE, GRID_SIZE, 2]

        # 모든 레이블과 그리드 셀에 대해 선과의 거리 계산
        # grid_centers를 [1, GRID_SIZE, GRID_SIZE, 2]로 확장
        grid_centers_expanded = tf.expand_dims(grid_centers, axis=0)  # [1, GRID_SIZE, GRID_SIZE, 2]
        # rho, cos_theta, sin_theta를 [num_labels, 1, 1, 1]로 확장
        rho_expanded = tf.reshape(rho, [-1, 1, 1, 1])  # [num_labels, 1, 1, 1]
        cos_theta_expanded = tf.reshape(cos_theta, [-1, 1, 1, 1])  # [num_labels, 1, 1, 1]
        sin_theta_expanded = tf.reshape(sin_theta, [-1, 1, 1, 1])  # [num_labels, 1, 1, 1]

        # 거리 계산
        x = grid_centers_expanded[..., 0]  # [1, GRID_SIZE, GRID_SIZE]
        y = grid_centers_expanded[..., 1]  # [1, GRID_SIZE, GRID_SIZE]

        distances = x * cos_theta_expanded + y * sin_theta_expanded - rho_expanded  # [num_labels, GRID_SIZE, GRID_SIZE]
        distances = tf.abs(distances)

        # 임계값 설정
        threshold = 1.0 / GRID_SIZE  # 스칼라

        # 마스크 생성
        mask = distances < threshold  # [num_labels, GRID_SIZE, GRID_SIZE]

        # 레이블 텐서 초기화
        label_tensor = tf.zeros((GRID_SIZE, GRID_SIZE, NUM_LINES, 3 + NUM_CLASSES), dtype=tf.float32)

        # 각 그리드 셀에 대해 처리
        for line_idx in range(NUM_LINES):
            # line_idx를 int64로 캐스팅
            line_idx = tf.cast(line_idx, tf.int64)

            # 현재 라인에 이미 레이블이 할당된 위치 확인
            existing_mask = label_tensor[..., line_idx, 0] > 0  # [GRID_SIZE, GRID_SIZE]

            # 현재 라인에 레이블이 비어있는 위치
            vacant_mask = tf.logical_not(existing_mask)  # [GRID_SIZE, GRID_SIZE]

            # 각 레이블에 대해 처리
            for label_idx in range(num_labels):
                # label_idx를 int64로 캐스팅
                label_idx = tf.cast(label_idx, tf.int64)

                # 해당 레이블의 마스크 추출
                label_mask = mask[label_idx]  # [GRID_SIZE, GRID_SIZE]

                # 빈 위치와 겹치는 부분 찾기
                update_mask = tf.logical_and(vacant_mask, label_mask)  # [GRID_SIZE, GRID_SIZE]

                # 업데이트할 위치의 인덱스 추출
                indices = tf.where(update_mask)  # [num_updates, 2], dtype=int64

                num_updates = tf.shape(indices)[0]
                if num_updates == 0:
                    continue

                # 업데이트할 값 생성
                updates = tf.tile([[1.0, rho[label_idx], theta[label_idx]]], [num_updates, 1])  # [num_updates, 3]

                # 인덱스에 라인 인덱스 추가
                line_indices = tf.fill([num_updates, 1], line_idx)  # [num_updates, 1], dtype=int64
                indices_full = tf.concat([indices, line_indices], axis=1)  # [num_updates, 3], dtype=int64

                # 업데이트할 채널 인덱스 생성 (0: 존재 여부, 1: rho, 2: theta)
                for channel_idx in range(3):
                    # channel_idx를 int64로 캐스팅
                    channel_idx = tf.cast(channel_idx, tf.int64)

                    channel_indices = tf.fill([num_updates, 1], channel_idx)  # [num_updates, 1], dtype=int64
                    indices_to_update = tf.concat([indices_full, channel_indices],
                                                  axis=1)  # [num_updates, 4], dtype=int64
                    updates_channel = updates[:, channel_idx]

                    # 레이블 텐서 업데이트
                    label_tensor = tf.tensor_scatter_nd_update(label_tensor, indices_to_update, updates_channel)

                # 업데이트된 위치를 vacant_mask에 반영
                vacant_updates = tf.fill([num_updates], False)
                vacant_mask = tf.tensor_scatter_nd_update(vacant_mask, indices, vacant_updates)

        return image, label_tensor

    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


model = create_line_detection_model()
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=line_detection_loss)

batch_size = 8
steps_per_epoch = 100
epochs = 10

data_dir = r'D:\Work\04_Samsung_Pyeongtak_WIND2\Data'

train_data = load_data(data_dir, batch_size)
model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs)


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


test_image_path = r'D:\Work\04_Samsung_Pyeongtak_WIND2\Data\FAIL_1.00_after.jpg'
image = tf.io.read_file(test_image_path)
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [INPUT_SIZE, INPUT_SIZE])
image = image / 255.0  # 정규화

detected_lines = inference(model, image)

print("검출된 직선 수:", len(detected_lines))
for line in detected_lines:
    print("직선 파라미터 (rho, theta):", line)
