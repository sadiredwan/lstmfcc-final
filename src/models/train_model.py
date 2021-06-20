import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import pickle
from tensorflow.keras import Sequential
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Flatten


def trial(hp):
	model = Sequential()
	model.add(LSTM(hp.Int('lstm_1', min_value=32, max_value=256, step=16), return_sequences=True, input_shape=X_train.shape[1:]))
	model.add(LSTM(hp.Int('lstm_2', min_value=32, max_value=256, step=16), return_sequences=True))
	model.add(LSTM(hp.Int('lstm_3', min_value=32, max_value=256, step=16), return_sequences=True))
	model.add(LSTM(hp.Int('lstm_4', min_value=32, max_value=256, step=16), return_sequences=True))
	model.add(TimeDistributed(Dense(hp.Int('tdd_1', min_value=16, max_value=128, step=16), activation='relu')))
	model.add(TimeDistributed(Dense(hp.Int('tdd_2', min_value=16, max_value=128, step=16), activation='relu')))
	model.add(TimeDistributed(Dense(hp.Int('tdd_3', min_value=16, max_value=128, step=16), activation='relu')))
	model.add(TimeDistributed(Dense(hp.Int('tdd_4', min_value=16, max_value=128, step=16), activation='relu')))
	model.add(Flatten())
	model.add(Dense(n_classes, activation='softmax'))
	model.compile(
		loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['acc'])
	return model


class RNN:
	def __init__(self, input_shape, output_shape, hyperparams):
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.lstm_1 = hyperparams['lstm_1']
		self.lstm_2 = hyperparams['lstm_2']
		self.lstm_3 = hyperparams['lstm_3']
		self.lstm_4 = hyperparams['lstm_4']
		self.tdd_1 = hyperparams['tdd_1']
		self.tdd_2 = hyperparams['tdd_2']
		self.tdd_3 = hyperparams['tdd_3']
		self.tdd_4 = hyperparams['tdd_4']


	def run(self):
		model = Sequential()
		model.add(LSTM(self.lstm_1, return_sequences=True, input_shape=self.input_shape))
		model.add(LSTM(self.lstm_2, return_sequences=True))
		model.add(LSTM(self.lstm_3, return_sequences=True))
		model.add(LSTM(self.lstm_4, return_sequences=True))
		model.add(TimeDistributed(Dense(self.tdd_1, activation='relu')))
		model.add(TimeDistributed(Dense(self.tdd_2, activation='relu')))
		model.add(TimeDistributed(Dense(self.tdd_3, activation='relu')))
		model.add(TimeDistributed(Dense(self.tdd_4, activation='relu')))
		model.add(Flatten())
		model.add(Dense(self.output_shape, activation='softmax'))
		model.summary()
		model.compile(
			loss='categorical_crossentropy',
			optimizer='adam',
			metrics=['acc'])
		return model


if __name__ == '__main__':
	os.chdir('../../')
	X = pickle.load(open('data/processed/acoustic/train/X_wpd_level1.pickle', 'rb'))
	y = pickle.load(open('data/processed/acoustic/train/y_train.pickle', 'rb'))
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=2)
	n_classes = len(y_train[0])

	LOG_DIR = 'models/logs/'+f'{int(time.time())}'

	tuner = RandomSearch(
		trial,
		objective='val_acc',
		max_trials=16,
		executions_per_trial=1,
		directory=LOG_DIR)

	tuner.search(
		x=X_train,
		y=y_train,
		epochs=1,
		batch_size=50,
		shuffle='true',
		validation_data=(X_val, y_val))

	model = RNN(
		input_shape=X_train.shape[1:],
		output_shape=n_classes,
		hyperparams=tuner.get_best_hyperparameters()[0].values).run()

	hist = model.fit(
		X_train,
		y_train,
		epochs=100,
		batch_size=50,
		shuffle='true',
		validation_data=(X_val, y_val))

	pickle.dump(hist.history, open('models/histories/wpd_mfcc.pickle', 'wb'))
	model.save('models/wpd_mfcc_model.h5')
