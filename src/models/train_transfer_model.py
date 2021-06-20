import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

os.chdir('../../')
pretrained_model = load_model('models/wpd_mfcc_model.h5', compile=True)
pretrained_model.trainable = False

for layer in pretrained_model.layers:
	if(layer.name.startswith('lstm')):
		layer.trainable = True


os.chdir('data/processed/throat')
X_train = pickle.load(open('train/X_wpd_level1.pickle', 'rb'))
y_train = pickle.load(open('train/y_train.pickle', 'rb'))
n_classes = len(y_train[0])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=2)

model = Sequential([pretrained_model, Dense(n_classes, activation='softmax')])

model.compile(
	loss='categorical_crossentropy',
	optimizer='adam',
	metrics=['acc'])

hist = model.fit(
	X_train,
	y_train,
	epochs=100,
	batch_size=50,
	shuffle='true',
	validation_data=(X_val, y_val))

os.chdir('../../../models')
pickle.dump(hist.history, open('histories/wpd_mfcc_throat_transfer.pickle', 'wb'))
model.save('wpd_mfcc_throat_transfer_model.h5')
