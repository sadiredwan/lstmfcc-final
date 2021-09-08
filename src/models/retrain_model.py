import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.models import clone_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


def reset_weights(model):
	for layer in model.layers: 
		if isinstance(layer, tf.keras.Model):
			reset_weights(layer)
			continue
		for k, initializer in layer.__dict__.items():
			if "initializer" not in k:
				continue
			var = getattr(layer, k.replace("_initializer", ""))
			var.assign(initializer(var.shape, var.dtype))


"""
Retrain model with or without pretrained weights
Args:
	model_name (acoustic_mfcc, acoustic_wpd1 etc.)
	training_mode ('transfer', 'new')
	data_set ('acoustic', 'throat', 'combined')
	decomposition_strategy ('raw', 'wpd', 'cwt')
	decomposition_level (1, 2, 3, 4)
Example:
	To train acoustic_wpd1 (keeping previous weights) with level 1 WPD coefficients of throat data
	python retrain_model.py transfer acoustic_wpd1 throat wpd 1
	To train acoustic_wpd1 (newly initialized weights) with level 1 WPD coefficients of combined data
	python retrain_model.py new acoustic_wpd1 combined wpd 1
"""
if __name__ == '__main__':
	os.chdir('../../')
	MODE = sys.argv[1]
	MODEL = sys.argv[2]
	DATASET = sys.argv[3]
	STRAT = sys.argv[4]
	
	pretrained_model = load_model('models/'+MODEL+'_model.h5', compile=True)
	
	if DATASET != 'combined':
		if STRAT != 'raw':
			LEVEL = sys.argv[5]
			X = pickle.load(open('data/processed/'+DATASET+'/train/X_'+STRAT+'_level'+LEVEL+'.pickle', 'rb'))
		else:
			X = pickle.load(open('data/processed/'+DATASET+'/train/X_raw.pickle', 'rb'))
		y = pickle.load(open('data/processed/'+DATASET+'/train/y_train.pickle', 'rb'))
	else:
		if STRAT != 'raw':
			LEVEL = sys.argv[5]
			X = np.vstack((pickle.load(open('data/processed/acoustic/train/X_'+STRAT+'_level'+LEVEL+'.pickle', 'rb')),
				pickle.load(open('data/processed/throat/train/X_'+STRAT+'_level'+LEVEL+'.pickle', 'rb'))))
		else:
			X = np.vstack((pickle.load(open('data/processed/acoustic/train/X_raw.pickle', 'rb')),
					pickle.load(open('data/processed/throat/train/X_raw.pickle', 'rb'))))
		y = np.vstack((pickle.load(open('data/processed/acoustic/train/y_train.pickle', 'rb')),
					pickle.load(open('data/processed/throat/train/y_train.pickle', 'rb'))))

	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=2)

	if MODE != 'transfer':
		model = clone_model(pretrained_model)
	else:
		pretrained_model.trainable = True
		model = pretrained_model

	model.compile(
	loss='categorical_crossentropy',
	optimizer='adam',
	metrics=['acc'])

	min_delta = (1e-7, 1e-3)[DATASET == 'throat']
	patience = (7, 3)[DATASET == 'throat']
	callback = EarlyStopping(
		monitor='val_loss',
		min_delta=min_delta,
		patience=patience,
		verbose=0,
		mode='auto',
		baseline=None,
		restore_best_weights=False)

	hist = model.fit(
	X_train,
	y_train,
	epochs=100,
	batch_size=50,
	shuffle='true',
	callbacks=[callback],
	validation_data=(X_val, y_val))

	if STRAT != 'raw':
		pickle.dump(hist.history, open('models/histories/'+DATASET+'_'+STRAT+LEVEL+'_'+MODE+'.pickle', 'wb'))
		model.save('models/'+DATASET+'_'+STRAT+LEVEL+'_'+MODE+'_model.h5')
	else:
		pickle.dump(hist.history, open('models/histories/'+DATASET+'_mfcc_'+MODE+'.pickle', 'wb'))
		model.save('models/retrained/'+DATASET+'_mfcc_'+MODE+'_model.h5')
