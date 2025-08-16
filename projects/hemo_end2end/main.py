import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from keras.layers import *
from tensorflow.keras.optimizers import Adam
import json
from qmap.benchmark import QMAPBenchmark
from qmap.toolkit.split import train_test_split
import math
import os
from pyutils import Colors

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

all_results = []
all_results_high_complexity = []
all_results_low_complexity = []
for split in range(5):
    for threshold in [55, 60]:
        benchmark = QMAPBenchmark(split, threshold,
                                  dataset_type='Hemolytic',
                                  forbidden_aa=['X', ' ']
                                  )
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

        test = np.array([encode_seq(seq) for seq in benchmark.inputs])
        preds = model.predict(test)
        results = benchmark.compute_metrics(preds)
        print(Colors.green, results, Colors.reset)
        all_results.append(results)

        high_comp_benchmark = benchmark.high_complexity
        preds = model.predict(np.array([encode_seq(seq) for seq in high_comp_benchmark.inputs]))
        all_results_high_complexity.append(high_comp_benchmark.compute_metrics(preds))

        low_comp_benchmark = benchmark.low_complexity
        preds = model.predict(np.array([encode_seq(seq) for seq in low_comp_benchmark.inputs]))
        all_results_low_complexity.append(low_comp_benchmark.compute_metrics(preds))



all_result_table = pd.DataFrame([all_result.dict() for all_result in all_results])
high_complexity = pd.DataFrame([result.dict() for result in all_results_high_complexity])
low_complexity = pd.DataFrame([result.dict() for result in all_results_low_complexity])

# Export to pandas
if not os.path.exists('results'):
    os.makedirs('results')
all_result_table.to_csv('results/full.csv')
high_complexity.to_csv('results/high_complexity.csv')
low_complexity.to_csv('results/low_complexity.csv')

print(all_results[0].md_col, end="")
for results in all_results:
    print(results.md_row, end="")