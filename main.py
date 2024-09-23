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

    # Backbone 네트워크 (예시로 간단한 CNN 사용)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(NUM_LINES * (1 + 2 + NUM_CLASSES), 1, padding='same')(x)
    x = tf.keras.layers.Reshape((GRID_SIZE, GRID_SIZE, NUM_LINES, 3 + NUM_CLASSES))(x)

    # 출력 분리
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

# 손실 함수 정의
def line_detection_loss(y_true, y_pred):
    """
    y_true와 y_pred의 형태:
    [batch_size, GRID_SIZE, GRID_SIZE, NUM_LINES, 3 + NUM_CLASSES]
    """
    # 직선 존재 여부 손실 (Binary Cross-Entropy)
    obj_mask = y_true[..., 0]
    confidence_loss = tf.keras.losses.binary_crossentropy(obj_mask, y_pred[..., 0])

    # 파라미터 손실
    # rho 손실 (Mean Squared Error)
    rho_loss = tf.square(y_true[..., 1] - y_pred[..., 1])

    # theta 손실 (주기성 고려)
    theta_diff = y_true[..., 2] - y_pred[..., 2]
    theta_diff = tf.math.floormod(theta_diff + np.pi, 2 * np.pi) - np.pi  # [-π, π] 범위로 조정
    theta_loss = tf.square(theta_diff)

    param_loss = rho_loss + theta_loss
    param_loss = obj_mask * param_loss  # 직선이 존재하는 곳만 계산

    # 총 손실 계산
    total_loss = tf.reduce_mean(confidence_loss + param_loss)
    return total_loss

# 데이터 로더 구현
def load_data(data_dir, batch_size):
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort()

    dataset = tf.data.Dataset.from_tensor_slices(image_files)

    def parse_function(filename):
        # 이미지 로드 및 전처리
        image_path = os.path.join(data_dir, filename)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [INPUT_SIZE, INPUT_SIZE])
        image = image / 255.0  # 정규화

        # 레이블 파일 로드
        label_filename = tf.strings.regex_replace(filename, '\.[^.]+$', '.txt')
        label_path = os.path.join(data_dir, label_filename)
        labels = tf.io.read_file(label_path)
        labels = tf.strings.strip(labels)
        labels = tf.strings.split(labels, '\n')
        labels = tf.strings.split(labels, ' ')
        labels = tf.strings.to_number(labels, out_type=tf.float32)
        labels = tf.reshape(labels, [-1, 2])  # [num_lines, 2]

        # 그리드 셀에 맞게 레이블 매핑
        label_tensor = tf.zeros((GRID_SIZE, GRID_SIZE, NUM_LINES, 3 + NUM_CLASSES), dtype=tf.float32)

        # 그리드 셀의 좌표 계산
        grid_x = tf.linspace(0.0, 1.0, GRID_SIZE + 1)
        grid_y = tf.linspace(0.0, 1.0, GRID_SIZE + 1)
        grid_x = (grid_x[:-1] + grid_x[1:]) / 2  # 그리드 셀의 중심 x 좌표
        grid_y = (grid_y[:-1] + grid_y[1:]) / 2  # 그리드 셀의 중심 y 좌표
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        grid_centers = tf.stack([grid_x, grid_y], axis=-1)  # [GRID_SIZE, GRID_SIZE, 2]

        for rho_theta in labels:
            rho = rho_theta[0]
            theta = rho_theta[1]

            # 선의 방정식: x * cos(theta) + y * sin(theta) = rho
            cos_theta = tf.math.cos(theta)
            sin_theta = tf.math.sin(theta)

            # 그리드 셀의 중심 좌표 대입하여 선과의 거리 계산
            distances = grid_centers[..., 0] * cos_theta + grid_centers[..., 1] * sin_theta - rho
            distances = tf.abs(distances)

            # 일정 임계값 이하인 그리드 셀을 선이 통과한다고 간주
            threshold = 1.0 / GRID_SIZE  # 그리드 셀 크기에 비례하여 임계값 설정
            mask = distances < threshold  # [GRID_SIZE, GRID_SIZE]

            # 레이블 텐서에 값 설정
            mask = tf.cast(mask, tf.float32)
            for i in range(NUM_LINES):
                existing_mask = label_tensor[..., i, 0]
                vacant = tf.cast(existing_mask == 0, tf.float32)
                update_mask = mask * vacant

                label_tensor = tf.tensor_scatter_nd_update(
                    label_tensor,
                    tf.where(update_mask > 0),
                    tf.tile([[1.0, rho, theta]], [tf.reduce_sum(tf.cast(update_mask > 0, tf.int32)), 1])
                )

        return image, label_tensor

    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset

# 모델 인스턴스 생성
model = create_line_detection_model()
model.summary()

# 옵티마이저 및 컴파일 설정
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=line_detection_loss)

# 학습 루프 실행
batch_size = 8
steps_per_epoch = 100
epochs = 10

# 데이터 디렉토리 지정
data_dir = 'path/to/data'

train_data = load_data(data_dir, batch_size)
model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs)

# 추론 예시
def inference(model, image):
    """
    단일 이미지에 대한 추론을 수행하고 예측된 직선을 반환합니다.
    """
    image = tf.expand_dims(image, axis=0)  # 배치 차원 추가
    preds = model.predict(image)

    # 직선 존재 확률이 임계값 이상인 예측만 선택
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

# 추론 테스트
test_image_path = 'path/to/test_image.jpg'  # 테스트할 이미지의 경로로 수정하세요
image = tf.io.read_file(test_image_path)
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [INPUT_SIZE, INPUT_SIZE])
image = image / 255.0  # 정규화

detected_lines = inference(model, image)

print("검출된 직선 수:", len(detected_lines))
for line in detected_lines:
    print("직선 파라미터 (rho, theta):", line)
