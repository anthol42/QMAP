import pandas as pd
import numpy as np
from tensorflow.keras import layers, callbacks
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential,Model
from keras.layers import *
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, matthews_corrcoef, average_precision_score,confusion_matrix
from sklearn.metrics import classification_report
from sklearn import model_selection, metrics
from sklearn.model_selection import train_test_split,StratifiedKFold,KFold
import json
from qmap.benchmark import QMAPBenchmark
from qmap.toolkit.split import train_test_split
import math
import os

tf.random.set_seed(0)
np.random.seed(0)

with open('../../data/build/dataset.json', 'r') as f:
    dataset = json.load(f)
    # Filter out sequences that are too long because the aligner support sequences up to 100 amino acids long
    dataset = [sample for sample in dataset if len(sample["Sequence"]) < 100]

    # Filter out D amino acids
    dataset = [sample for sample in dataset if not any(c.islower() for c in sample["Sequence"])]

    # Filter out X amino acids and ' '
    dataset = [sample for sample in dataset if 'X' not in sample["Sequence"] and ' ' not in sample["Sequence"]]

def encode_seq(seq):
    clas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    row = []
    for j in range(50):
        if j < len(seq):
            index = clas.index(seq[j])
            l = np.zeros(20)
            l[index] = 1
        else:
            l = np.zeros(20)
        row.append(l)
    return np.array(row)

def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(11, 11), activation='relu', padding='same', input_shape=(50, 20, 1)))
    model.add(Conv2D(32, kernel_size=(11, 11), activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=(7, 7), activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l2=0.0001)))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l2(l2=0.1)))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

if not os.path.exists('.cache'):
    os.makedirs('.cache')

for split in range(5):
    for threshold in [55, 60]:
        benchmark = QMAPBenchmark(split, threshold, dataset_type='Hemolytic')
        y = [sample['Hemolitic Activity'] for sample in dataset]
        X = np.stack([encode_seq(sample['Sequence']) for sample, target in zip(dataset, y) if not math.isnan(target)])
        X_sequences = [sample['Sequence'] for sample, target in zip(dataset, y) if not math.isnan(target)]
        y = np.array([target for target in y if not math.isnan(target)])

        train_mask = benchmark.get_train_mask(X_sequences, force_cpu=True)
        X_train = np.array([sample for is_valid, sample in zip(train_mask, X) if is_valid])
        X_sequences = [seq for is_valid, seq in zip(train_mask, X_sequences) if is_valid]
        y_train = np.array([target for is_valid, target in zip(train_mask, y) if is_valid])

        # Split dataset
        seq_train, seq_val, X_train, X_val, y_train, y_val = train_test_split(X_sequences, X_train, y_train, threshold=threshold / 100)

        model = get_model()
        callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                    tf.keras.callbacks.ModelCheckpoint(filepath='.cache/best_model1.h5', monitor='val_accuracy',
                                                       save_best_only=True, mode='auto')]
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                             callbacks=[callback], shuffle=True)

