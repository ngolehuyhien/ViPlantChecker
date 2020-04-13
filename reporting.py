from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

if __name__ == '__main__':
    # load json and create model
    json_file = open("model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("weights.h5")

    #Test set
    test_datagen= ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator=test_datagen.flow_from_directory('Report/',
                                            target_size=(224,224),
                                            color_mode='rgb',
                                            batch_size=32,
                                            class_mode='categorical',
                                            shuffle=False)

    # prepares data for confusion matrix
    Y_pred = model.predict_generator(test_generator, test_generator.n//test_generator.batch_size+1)
    y_pred = np.argmax(Y_pred, axis=1)

    print('===========================================')
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))
    target_names = ['Calotropis gigantea (L.) Dryand.', 'Achillea millefolium L.', 'Ricinus communis L.', 'Tamarindus indica L.', 'Punica granatum L.', 'Magnolia grandiflora L.', 'Reynoutria japonica Houtt.']
    print('===========================================')
    print('Classification Report')
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))