
## EnsembleCNN
## Modified from Paul Simonson's EnsembleCNN v1

import numpy as np
import pandas as pd
import time

import fcsparser

from sklearn.ensemble import RandomForestClassifier
import sklearn

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import tensorflow

from pathlib import Path
from joblib import dump, load
import os

strategy = tensorflow.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

class EnsembleCNNClassifier():

    def __init__(self):
        self.low_memory_approach = False

        self.name = 'EnsembleCNN'
        self.CNN_models = []
        self.output_folder = '../output/'
        self.models_save_folder = self.output_folder + self.name + '_models'
        self.num_features = 146
        self.num_bins = 25

        ## create model folder
        self.set_model_folder(self.models_save_folder)

        ## random forest
        self.RF = None
        self.n_estimators = 1000
        self.max_leaf_nodes = 256
        self.rand_seed = 1000
        self.init_RF(self.n_estimators, self.max_leaf_nodes, self.rand_seed)

        ## CNN
        self.num_classes = 9
        self.batch_size = 256
        self.epochs = 20
        self.validation_split = 0.2
        self.class_map = None

    def set_model_folder(self, models_save_folder):
        self.models_save_folder = models_save_folder
        if not os.path.exists(models_save_folder):
            os.makedirs(models_save_folder)

    def init_RF(self, n_estimators, max_leaf_nodes, rand_seed):
        self.n_estimators = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.rand_seed = rand_seed
        print('init RF..')
        self.RF = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, n_jobs=-1, verbose=1, class_weight='balanced', random_state = rand_seed)

    def init_CNN(self, input_dim, num_classes):
        self.num_classes = num_classes

        with strategy.scope():
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3,3), name='input_layer', activation = 'relu', padding='same',
                                    input_shape=(input_dim, input_dim, 1)))
            model.add(layers.Dropout(0.1))
            model.add(layers.Conv2D(32, (3,3), activation = 'relu', padding='same'))
            model.add(layers.Dropout(0.1))
            model.add(layers.MaxPooling2D((2,2)))
            model.add(layers.Conv2D(64, (3,3), activation = 'relu', padding='same'))
            model.add(layers.Dropout(0.1))
            model.add(layers.Conv2D(64, (3,3), activation = 'relu', padding='same'))
            model.add(layers.Dropout(0.1))
            model.add(layers.MaxPooling2D((2,2)))
            model.add(layers.Conv2D(64, (3,3), activation = 'relu', padding='same'))
            model.add(layers.Dropout(0.1))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, name='penultimate_layer', kernel_regularizer=regularizers.l2(0.001),
                                   activation='relu'))
            model.add(layers.Dropout(0.1))
            model.add(layers.Dense(num_classes, kernel_regularizer=regularizers.l2(0.001),
                                   activation='softmax'))
            model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

        return model

    def check_for_prior_models(self, models_save_folder):
        previously_completed_models = Path("").glob(pattern=str(models_save_folder) + "/" + "*.h5")

        restarting_fit = False
        start_with_model = len(list(previously_completed_models)) - 1
        if start_with_model < 0:
            start_with_model = 0
        else:
            restarting_fit = True
            print("Previously calculated CNN models are found in " +
                  models_save_folder + " and will be used. "
                  "Manually delete the *.h5 files if you want to recalculate them all."
                  "Now restarting fitting with model #" +
                  str(start_with_model + 1) +
                  " (deletes the last saved file, in case it was halted mid-save)...")
        return restarting_fit, start_with_model

    def fit_CNNs(self, X, y, batch_size = 256, epochs = 20, validation_split=0.2):
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split

        cat_classes = np.unique(y)
        self.class_map = dict(zip( list(cat_classes) , range(len(cat_classes))))
        y_int = [self.class_map[yi] for yi in y]

        classes = np.unique(y_int)
        num_classes = len(classes)
        self.num_classes = num_classes

        print(str(num_classes), "classes")
        print(str(self.num_bins), "bin size")

        # le = sklearn.preprocessing.LabelEncoder()
        # y_int = le.fit_transform(y)

        num_columns = self.num_bins * self.num_bins

        class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_int), y=y_int)
        class_weights = dict(enumerate(class_weights))

        restarting_fit, start_with_model = self.check_for_prior_models(self.models_save_folder)

        for i in range(start_with_model, self.num_features):

            print("======== training model", str(i), "==========")

            #Iterate over the features (number of 2D histograms...).
            #Create the array for holding the resuling 2d heatmaps for all the files, one row per file.
            data_array = X[:, (i*num_columns):((i+1)*num_columns)]
            data_array = data_array.reshape((len(y_int), self.num_bins,
                                             self.num_bins, 1)).astype('float32')
            #Make the labels categorical:
            y_cat = tensorflow.keras.utils.to_categorical(y_int)

            ## new CNN
            model = self.init_CNN(self.num_bins, num_classes)
            print(model.summary())

            history = model.fit(data_array, y_cat, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose = 1, class_weight=class_weights)

            print(history)

            model_filepath = self.models_save_folder + '/' + str(i+1) + '.h5'
            model.save(model_filepath)
            print('Saved ' + model_filepath)

            self.CNN_models.append(model)

    def reload_CNNs(self):
        for i in range(0, self.num_features):
            model_filepath = self.models_save_folder + '/' + str(i+1) + '.h5'
            model = models.load_model(Path(model_filepath))
            self.CNN_models.append(model)
            print("loaded" + model_filepath)
        return None

    def get_CNN_outputs(self, X):
        num_columns = num_columns = self.num_bins * self.num_bins
        prediction_values_matrix = np.zeros((X.shape[0], self.num_features))

        data_array = []
        prediction_values = []
        for i in range(0, self.num_features):
            #Create the array for holding the resuling 2d heatmaps for all the files, one row per file.
            data_array = X[:, (i*num_columns):((i+1)*num_columns)]
            data_array = data_array.reshape((X.shape[0], self.num_bins, self.num_bins, 1)).astype('float32')

            prediction_values = self.CNN_models[i].predict(data_array)
            prediction_values_matrix[:,i] = prediction_values[:,1]

            #Make sure to use the next model...
            #trained_model_number = trained_model_number + 1
            K.clear_session()

        return prediction_values_matrix

    def fit_RF(self, X, y, n_estimators, max_leaf_nodes, rand_seed):
        self.n_estimators = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.rand_seed = rand_seed
        self.init_RF(n_estimators, max_leaf_nodes, rand_seed)
        self.RF.fit(X, y)
        return None

    def get_RF_outputs(self, X):
        return self.RF.predict(X)

    def run_EnsembleCNN(self, X_train, y_train, X_test, y_test, train_CNN_RF_split=0.5):
        return None
