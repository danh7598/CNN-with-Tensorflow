import tensorflow as tf
from keras.utils import np_utils
import cv2

# Lấy dữ liệu train từ thư viện, dữ liệu chữ số viết tay
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print x_train.shape = (60000, 28, 28)
# print y_train.shape = (60000,)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # Chuyển shape về (60000, 28, 28, 1)
x_train = x_train.astype('float32')  # Ép kiểu float 32bit
x_train /= 255  # Vì ánh xạ xác suất từ 0 đến 1, nên quy về từ 0 đến 1 cho 0 đến 255
y_train = np_utils.to_categorical(y_train, 10)  # Chuyển thành vector one hot

# Mạng CNN
model = tf.keras.models.Sequential([
    # Số bộ lọc 32, kích thước bộ lọc 3x3, kích thước input = 28x28 vì hình train là 28x28
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1),
                           activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Lớp max pooling 2 chiều có kích thước 2x2
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),  # Lặp lại thêm một lớp convo
    # để tăng độ phức tạp của mạng
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),  # Lớp làm phẳng ma trận
    tf.keras.layers.Dense(300, activation=tf.nn.relu),  # Lớp Fully Connected với 300 neuron
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),  # Lớp Fully Connected output với 10 neuron đầu ra
    # hàm kích hoạt softmax để ánh xạ xác suất
])

# Cấu hình lost function, update weight
# loss function: trung bình bình phương lỗi, update weight bằng gradient descent
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
# model.summary()

# Train model
# [1, 2, 3, 4]
# 1: số lượng sample
# 2, 3: kích thước ma trận
# 4: số chiều, ảnh màu: 3, ảnh xám: 1
model.fit(x_train, y_train, epochs=30)
model.save('model.h5') # Lưu model dùng lần sau
img_test = cv2.imread('anh4.png', 0)  # Đọc ảnh 3 dưới dạng ảnh xám
cv2.imshow('anh test', img_test)  # Hiển thị hình ảnh
# cv2.waitKey(0)  # Chờ nhấn phím thì mới thoát
# print img_test.shape = (28, 28)
img_test = img_test.reshape(1, 28, 28, 1)  # reshape ảnh về ma trận 4 chiều cần thiết
print(model.predict(img_test)) # Dự đoán ảnh
