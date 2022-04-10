import numpy as np
import cv2
import pickle
import tensorflow as tf

width = 200
height = 200

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
local_host_load_option = tf.saved_model.LoadOptions(experimental_io_device="/job:localhost")
# pickle_in = open("model_trained.p", "rb")
# model = pickle.load(pickle_in)
model = tf.keras.models.load_model(r"C:/Users/subha/PycharmProjects/pythonProject2",
                                   options=local_host_load_option)


def pre_processing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = image / 255
    return image


while True:
    success, imgOriginal = cap.read()
    # print(success, imgOriginal)
    img = np.array(imgOriginal)
    # print(img)
    img = cv2.resize(img, (32, 32))
    img = pre_processing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    # classIndex = int(model.predict_classes(img))
    # print(classIndex)
    prediction = model.predict(img)
    prediction_no = prediction.argmax(axis=-1)
    probability = np.amax(prediction)
    # probability = round(probability, 3)
    # prediction = (model.predict(img) > 0.4).astype("int32")
    # print(prediction)
    # print_text = ("%d %2.2f %" %(prediction_no, probability*100))
    print_text = str(prediction_no)+" "+str(round(probability, 2)*100)+"%"
    if probability > 0.4:
        cv2.putText(imgOriginal, print_text,
                    (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    # print(prediction_no, probability)
    cv2.imshow("Original Image", imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
