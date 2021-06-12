import tensorflow as tf
import numpy as np
import pandas as pd
print(tf.__version__)
import tensorflow.keras.layers as tfl
from tensorflow import keras
import scipy
import h5py
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from datetime import datetime

np.random.seed(1)
print("All packages imported!")

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

Y_train = np.array(train_data.label)
Y_train = Y_train.reshape(Y_train.shape[0],1)
Y_train = convert_to_one_hot(Y_train, 10).T
print(Y_train.shape)

X_train = train_data
X_train.drop(['label'],axis=1,inplace=True)
X_train = np.array(X_train)
print(X_train.shape)

X_train = np.reshape(X_train, (X_train.shape[0],-1, 28,1))
print(X_train.shape)

index = 3679
plt.imshow(X_train[index]) #display sample training image
plt.show()
print(Y_train[index])

X_test = test_data
X_test = np.array(X_test)
print(X_test.shape)

X_test = np.reshape(X_test, (X_test.shape[0],-1, 28,1))
print(X_test.shape)


ind = []
for i in range(X_test.shape[0]):
    ind.append(i+1)

test_data['ImageId'] = ind
print(test_data.head())

def convnet(inp_shape):
    #hyperparameters
    stride1 =1
    maxpool_stride1 = 2
    maxpool_size1 = 8
    no_f1 = 32
    f1 = 5

    stride2 =1
    maxpool_stride2 = 2
    maxpool_size2 = 8
    no_f2 = 16
    f2 = 2

    #Model definition

    input_img = tf.keras.Input(shape = inp_shape)

    Z1 = tfl.Conv2D(no_f1, (f1, f1), stride1, padding='same')(input_img)
    A1 = tfl.ReLU()(Z1)
    P1 = tfl.MaxPooling2D(pool_size=maxpool_size1, strides=maxpool_stride1, padding='same')(A1)
    Z2 = tfl.Conv2D(no_f2, (f2, f2), stride2, padding='same')(P1)
    A2 = tfl.ReLU()(Z2)
    P2 = tfl.MaxPooling2D(pool_size=maxpool_size2, strides=maxpool_stride2, padding='same')(A2)
    Z3 = tfl.Conv2D(no_f1, (f1, f1), stride1, padding='same')(P2)
    A3 = tfl.ReLU()(Z3)
    P3 = tfl.MaxPooling2D(pool_size=maxpool_size1, strides=maxpool_stride1, padding='same')(A3)
    Z4 = tfl.Conv2D(no_f2, (f2, f2), stride2, padding='same')(P3)
    A4 = tfl.ReLU()(Z4)
    P4 = tfl.MaxPooling2D(pool_size=maxpool_size2, strides=maxpool_stride2, padding='same')(A4)
    F = tfl.Flatten()(P4)
    outputs = tfl.Dense(10, activation="softmax")(F)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    return model

conv_model = convnet((28,28,1))
conv_model.summary()

now1 = datetime.now().time() # time object1
conv_model.fit(X_train,Y_train,batch_size=64,epochs=5)
now2 = datetime.now().time() # time object2

print("Training Start Time =", now1)
print("Training End Time =", now2)

conv_model.save("4_layer_5_epoch_model")

conv_model = keras.models.load_model("4_layer_5_epoch_model")
X_test = X_test.astype("float32")
now1 = datetime.now().time() # time object1
predictions = conv_model.predict(X_test)
now2 = datetime.now().time() # time object2
no_predictions = np.argmax(predictions, axis=1)
print(predictions.shape)
print(no_predictions)

#checking prediction
for index in range(2678,2690):
    plt.imshow(X_test[index]) #display predicted number and image
    plt.show()
    print(no_predictions[index])

print("Prediction Start Time =", now1)
print("Prediction End Time =", now2)

output = pd.DataFrame({'ImageId': test_data.ImageId, 'Label': no_predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")