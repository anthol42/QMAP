import keras
from keras.layers import Dense, Dropout, LSTM, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, Flatten, ZeroPadding1D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import json
from keras.models import model_from_json
import numpy as np
from src.settings import MAX_SEQUENCE_LENGTH, MAX_MIC, character_to_index, max_mic_buffer
from scipy.stats import kendalltau

def conv_model():
    model = keras.models.Sequential()
    model.add(ZeroPadding1D(
        5, input_shape = (MAX_SEQUENCE_LENGTH, len(character_to_index) + 1)
    ))
    model.add(Conv1D(
        64,
        kernel_size = 5,
        strides = 1,
        activation = 'relu',
        #input_shape = (MAX_SEQUENCE_LENGTH, len(character_to_index) + 1)
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    #model.add(Dropout(0.5))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    #model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def evaluate(model, test_x, test_y):
    predictions = model.predict(test_x)
    correctly_classified_error = np.mean([
        (actual - predicted) ** 2 
        for actual, predicted in zip(test_y, predictions)
        if actual < MAX_MIC and predicted < MAX_MIC - max_mic_buffer
    ])    
    all_error = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(test_y, predictions)])    
    all_active_error = np.mean([
        (actual - predicted) ** 2
        for actual, predicted in zip(test_y, predictions)
        if actual < MAX_MIC
    ])    
    return correctly_classified_error, all_active_error, all_error

def evaluate_model(model, test_x, test_y):
    predicted = model.predict(test_x)
    predicted = [pred[0] for pred in predicted]
#     plt.plot(predicted,test_y,'.')
#     plt.show()
    to_return = 'RMSE,'+repr(np.sqrt(float(np.average([(predicted[i]-test_y[i])**2 for i in range(len(test_y))]))))
    to_return += ',Pearson Correlation,'+repr(float(np.corrcoef(predicted,test_y)[1,0]))
    tau,pval = kendalltau(predicted,test_y)
    to_return += ',Kendall tau Correlation,'+repr(tau)
    to_return += ',MSE,'+repr(float(np.average([(predicted[i]-test_y[i])**2 for i in range(len(test_y))])))
    return to_return

def evaluate_as_classifier(model, test_x, test_y, debug=False):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    all_predicted = model.predict(test_x)
    for i in range(len(test_y)):
        actual = test_y[i]
        predicted = all_predicted[i]
        if actual < MAX_MIC - 0.0001:
            if predicted < MAX_MIC - max_mic_buffer:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if predicted < MAX_MIC - max_mic_buffer:
                false_positives += 1
                if debug == True:
                    print(vector_to_amp(test_x[i]))
                    print('predicted: ' + repr(predicted) + ', actual: '+repr(actual))
                    print('>p' + repr(false_positives) + '_' + repr(predicted))
                    print(vector_to_amp(test_x[i])['sequence'].replace('_', ''))
            else:
                true_negatives += 1
    return true_positives, true_negatives, false_positives, false_negatives

