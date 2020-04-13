from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import cv2
import math

if __name__ == '__main__':
    # load json and create model
    json_file = open("model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights("weights.h5")

    # load image
    img = cv2.imread('Test/Magnolia grandiflora L.jpg')
    img = cv2.resize(img,(224,224))
    img = np.reshape(img,[1,224,224,3])

    # predict
    target_names = ['Calotropis gigantea (L.) Dryand.', 'Achillea millefolium L.', 'Ricinus communis L.', 'Tamarindus indica L.', 'Punica granatum L.', 'Magnolia grandiflora L.', 'Reynoutria japonica Houtt.']
    Y_pred = model.predict(img)
    print(Y_pred)
    y_pred = np.argmax(Y_pred, axis=1)

    # print
    print('===========================================')
    print('Recognised Plant Species: ', target_names[y_pred[0]])
    # print('Recognised Plant Species: ', decode_recognized_species(y_pred[0]))