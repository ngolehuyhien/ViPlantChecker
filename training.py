import tensorflow as tf
import keras

from keras.layers.core import Dense, Activation
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.applications import MobileNetV2
from keras.models import Model
from keras.models import Sequential
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input

if __name__ == '__main__':
    # Configure the session to avoid reserving all available memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # Create the base model and add some extra layers to adjust to our model
    base_model=keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) # add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) # dense layer 2
    x=Dense(512,activation='relu')(x) # dense layer 3
    preds=Dense(7,activation='softmax')(x) # final layer with softmax activation

    model=Model(inputs=base_model.input,outputs=preds)

    # Setting how many layers will be trained
    for layer in model.layers[:20]:
        layer.trainable=False
    for layer in model.layers[20:]:
        layer.trainable=True

    # Training set
    train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator=train_datagen.flow_from_directory('Training/',
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)
    # Validation set
    val_datagen= ImageDataGenerator(preprocessing_function=preprocess_input)
    val_generator=val_datagen.flow_from_directory('Validation/',
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)

    # Compile and train model
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

    step_size_train=train_generator.n//train_generator.batch_size
    step_size_val=val_generator.n//val_generator.batch_size
    model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=10, validation_data=val_generator,
                    validation_steps=step_size_val)

    # Save the model
    model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    # json_file.write(model_json)
    model.save_weights("weights.h5") # serialize weights to HDF5