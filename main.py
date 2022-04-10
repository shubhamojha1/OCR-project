import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
# import pydot
# import graphviz
# from ann_visualizer.visualize import ann_viz
import matplotlib.pyplot as plt
import pickle

###################################################
path = 'myData'
testRatio = 0.2
validationRatio = 0.2

###################################################
images = []
classNo = []
myList = os.listdir(path)
# print(len(myList))
noOfClasses = len(myList)

for x in range(noOfClasses):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        currImg = cv2.imread(path + "/" + str(x) + "/" + y)
        currImg = cv2.resize(currImg, (32, 32))
        images.append(currImg)
        classNo.append(x)

#     print(x)           # 0 ... 9
# print(len(images))     # 10160
# print(len(classNo))    # 10160

images = np.array(images)
classNo = np.array(classNo)

# print(images.shape)     # (10160, 32, 32, 3)
# print(classNo.shape)    # (10160,)


# SPLITTING DATA
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (8128, 32, 32, 3) (2032, 32, 32, 3) (8128,) (2032,)

x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validationRatio)
# print(x_train.shape, x_validation.shape, y_train.shape, y_validation.shape)
# (6502, 32, 32, 3) (1626, 32, 32, 3) (6502,) (1626,)

numOfSamples = []
for x in range(noOfClasses):
    numOfSamples.append(len(np.where(y_train == x)[0]))


# print(numOfSamples)
# plt.figure(figsize=(10, 5))
# plt.bar(range(noOfClasses), numOfSamples)
# plt.title("No of images for each class")
# plt.xlabel("Class ID")
# plt.ylabel("No. of Images")
# plt.show()


def pre_processing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = image / 255
    return image


# Testing above function
# img = pre_processing(x_train[69])
# img = cv2.resize(img, (300, 300))
# cv2.imshow("Preprocessed Image", img)
# cv2.waitKey(0)


x_train = np.array(list(map(pre_processing, x_train)))
x_test = np.array(list(map(pre_processing, x_test)))
x_validation = np.array(list(map(pre_processing, x_validation)))

# Adding depth to images, so that CNN runs them properly
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)

"""
augmenting, zoom, rotation, translation
to make data more generic and better at predictions
"""

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

dataGen.fit(x_train)
"""
ONE HOT ENCODING
"""
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

image_dimensions = (32, 32, 3)


def my_model():
    no_of_filters = 60
    size_of_filter1 = (5, 5)
    size_of_filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_of_nodes = 500

    model = Sequential()

    model.add((Conv2D(no_of_filters,
                      size_of_filter1,
                      input_shape=(32, 32, 1),
                      activation='relu')))
    model.add((Conv2D(no_of_filters,
                      size_of_filter1,
                      activation='relu')))

    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add((Conv2D(no_of_filters // 2,
                      size_of_filter2,
                      activation='relu')))
    model.add((Conv2D(no_of_filters // 2,
                      size_of_filter2,
                      activation='relu')))

    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(no_of_nodes, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(noOfClasses, activation='softmax'))

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


model = my_model()
# print(model.summary())
# ann_viz(model, title="My first neural network")
# tf.keras.utils.plot_model(model, to_file='model.png',
#                           show_shapes=False,show_dtype=False,
#                           show_layer_names=True,
#                           expand_nested=False,
#                           dpi=96,
#                           show_layer_activations=True)
batch_size_value = 50
no_of_epochs = 10
steps_per_epoch = 130
validation_steps = 40
history = model.fit(dataGen.flow(x_train, y_train, batch_size=batch_size_value),
                    steps_per_epoch=steps_per_epoch,
                    epochs=no_of_epochs,
                    validation_data=(x_validation, y_validation),
                    validation_steps=validation_steps,
                    shuffle=True)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.show()
plt.savefig("loss_graph.png")

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
plt.savefig("accuracy_graph.png")

score = model.evaluate(x_test, y_test, verbose=0)

print("Test Score: ", score[0])
print("Test Accuracy: ", score[1])

# pickle_out = open("model_trained.p", "wb")
# pickle.dump(model, pickle_out)
# pickle_out.close()
local_host_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
model.save(os.getcwd(), options=local_host_save_option)
