import tensorflow as tf
import cv2

model = tf.keras.models.load_model('model.h5')  # load model

img_test = cv2.imread('anh9.png', 0)  # Đọc ảnh xám
img_test = img_test.reshape(1, 28, 28, 1)
print(model.predict(img_test))
