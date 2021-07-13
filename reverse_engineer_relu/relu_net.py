import tensorflow as tf
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import KFold
import os


def load_dataset():
    (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    train = tf.keras.utils.to_categorical(trainY)
    testY = tf.keras.utils.to_categorical(testY)

    return trainX, trainY, testX, testY

def prep_pixels(train,test):
    train_norm = train.astype("float32")
    test_norm = test.astype("float32")

    train_norm = train_norm/255.0
    test_norm = test_norm/255.0
    return test_norm, train_norm


def model_def():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer = opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(dataX):
        model = model_def()
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        history = model.fit(trainX, trainY, epochs=5, batch_size=30, validation_data=(testX, testY), verbose=0)
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        scores.append(acc)
        histories.append(history)
    
    model_json = model.to_json()
    with open("model1.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model1_weights.h5")
    return scores, histories
 
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(2, 1, 1)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(2, 1, 2)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	pyplot.show()

def summarize_performance(scores):
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    pyplot.boxplot(scores)
    pyplot.show()



def run_test():
	trainX, trainY, testX, testY = load_dataset()
	trainX, testX = prep_pixels(trainX, testX)
	scores, histories = evaluate_model(trainX, trainY)
	# summarize_diagnostics(histories)
	# summarize_performance(scores)




run_test()



