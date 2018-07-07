import os
import numpy
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from numpy import array
from PIL import Image
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from matplotlib import pyplot as plt
from keras import optimizers
from sklearn import preprocessing
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
import matplotlib.image as mpimg

def LoadData(path):
	contents = []
	with open(path) as f:
		contents = f.readlines()

	emotion = []
	pixel = []
	usage = []

	for i in contents:
		e,p,u = i.split(',')
		emotion.append(float(e))

		p = p.split(' ')
		p = numpy.array(p)
		p = p.astype(numpy.float)
		p = p.reshape(48,48)
		pixel.append(p)

		
		if u == 'Training\n':
			usage.append(1)
		else:
			usage.append(0)

	emotion = numpy.array(emotion)
	pixel = numpy.array(pixel)

	return emotion,pixel,usage

def ReShape(pix,dim):
	data=[]
	for i in pix:
		i = numpy.array(i)
		i = i.reshape(dim,dim)
		data.append(i)
	return data

def Evaluate(model):
	emotion,pixel,usage = LoadData('test.csv')
	pixel = pixel.reshape(200, 48,48, 1)
	pixel = pixel.astype('float32')
	pixel = pixel/255.
	emotion = keras.utils.to_categorical(emotion,4)

	print('***********************************************************')
	ic = model.predict_classes(pixel)
	test_eval = model.evaluate(pixel, emotion, verbose=0)
	
	print('Test Loss: ' + str(test_eval[0]))
	print('Test Acc: ' + str(test_eval[1]))

	for data in zip(emotion,ic):
		print(data)
	print('***********************************************************')

def CreateModel():
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(48,48,1)))
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D((2, 2),padding='same'))
	model.add(Dropout(0.25))
	model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	model.add(Dropout(0.25))
	model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
	model.add(LeakyReLU(alpha=0.1))                  
	model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(128, activation='linear'))
	model.add(LeakyReLU(alpha=0.1))           
	model.add(Dropout(0.3))
	model.add(Dense(4, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

	return model
		
emotion,pixel,usage = LoadData('data.csv')
pixel = pixel.reshape(-1, 48,48, 1)
pixel = pixel.astype('float32')
pixel = pixel/255.
emotion = keras.utils.to_categorical(emotion,4)

print(pixel.shape)
print(emotion.shape)

model = CreateModel()

#model = load_model('facial.h5py')

Evaluate(model)
for i in range(1,51):
	print("***************************** Training: " + str(i) + " *******************************")
	model_train = model.fit(pixel, emotion, batch_size=64,epochs=1,verbose=1)
	model.save("facial.h5py")


img = mpimg.imread('/home/mohan/Pictures/angry.jpg')
img = img/255.
img = img.reshape(-1,48,48,1)
res = model.predict_classes(img)
print(res)
               
