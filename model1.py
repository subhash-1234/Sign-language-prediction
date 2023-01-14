import cv2
import tensorflow as tf

CATEGORIES = ["1", "2","3","4"]  # will use this to convert prediction num to string value


def prepare(filepath):
    IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("64x3-CNN1.model")
prediction = model.predict([prepare('/home/subhash/Documents/project/2/2.6e606d22-9eac-11eb-8737-acde48001122.jpg')])
print(CATEGORIES[int(prediction)])