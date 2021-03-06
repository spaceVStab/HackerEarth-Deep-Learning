#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 17:47:50 2017

@author: ajinkya
"""
'''
TO DO
 1) check in  add_new_last_layer() whether to include GlobalAveragePooling2D or not.{
 __main__:47: UserWarning: Update your `Model` call to the Keras 2 API: `Model(inputs=Tensor("in..., outputs=Tensor("de...)`
 }

 2) confusion between sample_per_epoch and batch_size

3) Look around early stopping parameter (callback)
'''
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


#from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception, preprocess_input

from keras.preprocessing.image import ImageDataGenerator 

import matplotlib.pyplot as plt

def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x)
  x = BatchNormalization()(x)#new FC layer, random init
  x = Dropout(0.6,seed=8)(x)
#  x = Dense(int(FC_SIZE/2), activation='relu')(x)
#  x = Dropout(0.5,seed=8)(x)
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model


def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
  Args:
    model: keras model
  """
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])

def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')
  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()

####### Training Part
IM_WIDTH, IM_HEIGHT = 299, 299
batch_size=32
nb_epoch=4
nb_train_samples=2919
nb_val_samples=296
FC_SIZE = 512
NB_IV3_LAYERS_TO_FREEZE = 96



train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=40,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,)

train_generator = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(IM_WIDTH, IM_HEIGHT),
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical',
                                                 seed=8)

validation_generator = validation_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(IM_WIDTH, IM_HEIGHT),
                                            batch_size = 64,
                                            class_mode = 'categorical',
                                            seed=8,
                                            shuffle=False)

  # setup model
base_model = Xception(weights='imagenet', include_top=False) #include_top=False excludes final FC layer

#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)

model = add_new_last_layer(base_model, nb_classes=25)
    
      # transfer learning
setup_to_transfer_learn(model, base_model)

early_stop=EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')
    
history_tl = model.fit_generator(
        train_generator,
        epochs=4,
        steps_per_epoch=int(nb_train_samples/batch_size),
        validation_data=validation_generator,
#        callbacks=[early_stop],
        validation_steps=int(nb_val_samples/64)
        )
    
      # fine-tuning
setup_to_finetune(model)
    
history_ft = model.fit_generator(
        train_generator,
        steps_per_epoch=int(nb_train_samples/batch_size),
        epochs=nb_epoch,
        validation_data=validation_generator,
        callbacks=[early_stop],
        validation_steps=int(nb_val_samples/64)
        )
    
#model.save("inceptionv3-ft_1_228.model")
model.save("xception_96.model")

    
plot_training(history_ft)