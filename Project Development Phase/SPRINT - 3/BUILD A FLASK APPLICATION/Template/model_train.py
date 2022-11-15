from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten

#image loading and resizing
train_datagen= ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen= ImageDataGenerator(rescale = 1./255)
x_train= train_datagen.flow_from_directory('dataset/training_set', target_size=(64,64),batch_size=300,class_mode='categorical',color_mode="grayscale")
x_test= train_datagen.flow_from_directory('dataset/test_set', target_size=(64,64),batch_size=300,class_mode='categorical',color_mode="grayscale")

#Creating the model
model=Sequential()
model.add(Convolution2D(32,(3,3), input_shape=(64,64,1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=512,activation='relu'))
model.add(Dense(units=9,activation='softmax'))
model.compile(loss='categorical_crossentropy')
model.fit_generator(x_train, steps_per_epoch=24,epochs=10, validation_data=x_test,validation_steps=40)
model.save('aslpng1.h5')
