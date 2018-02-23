import csv
import cv2
import numpy as np
lines = []
with open('driving_log.csv') as csvdata:
	reader = csv.reader(csvdata)
	
	for line in reader:
		lines.append(line)
		
images = []
measurements = []
for line in lines:
	path = line[0]
	file_name = path.split('\\')[-1]
	curr_path = 'IMG\\'+file_name
	img = cv2.imread(curr_path, 1)
	images.append(img)
	measurement = float(line[3])
	measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3) ))
model.add(Convolution2D(64,3,3))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout(0.6)
# model.add(Convolution2D(32,3,3))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(128,3,3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(150))
model.add(Dropout(0.5))
model.add(Activation('softmax'))
model.add(Dense(80))
model.add(Activation('softmax'))
model.add(Dense(1))
model.add(Activation('softmax'))
# model.add(Flatten(input_shape=(160,320,3)))
# model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
model.fit(X_train, y_train, validation_split = 0.3, shuffle = True, nb_epoch =3)

model.save('model.h5')