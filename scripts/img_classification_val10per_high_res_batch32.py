##############
# set validation_split=0.1 to see if accuarcy is improved
# use epochs = 15 as previous model shows overfitting after epochs =10
##############

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from PIL import ImageFile
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from ann_visualizer.visualize import ann_viz

# set this as using tensorflow as backend
K.set_image_dim_ordering('th')
# avoid IOError: image file is truncated (xx bytes not processed)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# set variables
train_data_dir = ' '
img_height = 224
img_width = 224
batch_size = 32
epochs = 50

# load image data
train_datagen = ImageDataGenerator( rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    validation_split=0.1) # set validation split as 0.1 to get more data to train the model

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(train_data_dir, # same directory as training data
                                                         target_size=(img_height, img_width),
                                                         batch_size=batch_size,
                                                         class_mode='binary',
                                                         subset='validation') # set as validation data

# build deep learning models

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, img_height, img_width)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', #compile model for ternary img classfication
              optimizer='adam',
              metrics=['accuracy'])


# fit the model
history = model.fit_generator(train_generator,
                              steps_per_epoch=2000 // batch_size,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=800 // batch_size
                             )

# set model index
model_index = '11th_batch32'

# serialize model to JSON
model_json = model.to_json()
with open('{}_try_img_classfication.json'.format(model_index), "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights('{}_try_img_classfication.h5'.format(model_index))  
print("Saved model to disk")

model.save_weights('{}_try_img_classfication.h5'.format(model_index))  

# save callback history:
f = open('{}_model_history.pckl'.format(model_index), 'wb')
pickle.dump(history.history, f)
f.close()

# generate figures for model loss and accuracy
objects = []
file_dir = "/home/ec2-user/SageMaker/mengzhao/image_classification/scripts/{}_model_history.pckl"

# load CNN history containing loss and accuracy of each porch
with (open(file_dir.format(model_index), "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

# contruct pandas dataframe
objects_df = pd.DataFrame({'train_accuracy': objects[0]['acc'],
                           'validation_accuracy': objects[0]['val_acc'],
                           'train_loss': objects[0]['loss'],
                           'validation_loss': objects[0]['val_loss']}
                         )
fontsize = 12

# plot model's accuracy
objects_df[['train_accuracy','validation_accuracy']].plot()
plt.title('model accuracy', fontsize = fontsize)
plt.ylabel('accuracy', fontsize = fontsize)
plt.xlabel('epoch', fontsize = fontsize)
plt.savefig("{}_model_accuray.png".format(model_index), dpi = 400)

# plot model's loss
objects_df[['train_loss','validation_loss']].plot()
plt.title('model loss', fontsize = fontsize)
plt.ylabel('loss', fontsize = fontsize)
plt.xlabel('epoch', fontsize = fontsize)
plt.savefig("{}_model_loss.png".format(model_index), dpi = 400)
