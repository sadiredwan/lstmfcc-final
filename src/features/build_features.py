import os
import gc
import sys
import pywt
import random
import pickle
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
from python_speech_features import mfcc


def get_noise(path):
	background_noise = []
	background = [f for f in os.listdir(
		join(path, '_background_noise_')) if f.endswith('.wav')]
	for wav in background:
		sample, rate = librosa.load(
			join(join(path, '_background_noise_'), wav))
		background_noise.append(sample)
	return background_noise


def augment_silence(background_noise, config, n):
	silence = []
	num_noise = n//len(background_noise)
	for i, _ in enumerate(background_noise):
		for _ in range(num_noise):
			silence.append(__noise__(background_noise, i, config.samplerate))
	silence = np.array(silence)
	return silence


def __noise__(background_noise, idx, sr):
	noise = background_noise[idx]
	start_idx = random.randint(0, len(noise) - sr)
	return noise[start_idx:(start_idx + sr)]


def grab_files(path, ext='wav'):
	labels, filenames = [], []
	for ff in os.listdir(path):
		if os.path.isdir(join(path+ff)):
			for f in os.listdir(join(path+ff)):
				if f.endswith(ext):
					labels.append(ff)
					filenames.append(f)
	return labels, filenames


def label_transform(labels):
	n_labels = []
	legal_labels = 'down go left no off on right silence stop unknown up yes'.split()
	for label in labels:
		if label == '_background_noise_':
			n_labels.append('silence')
		elif label not in legal_labels:
			n_labels.append('unknown')
		else:
			n_labels.append(label)
	return pd.get_dummies(pd.Series(n_labels))


"""Returns WPD coefficients of wav signal
Parameters
----------
signal : numpy ndarray
	Input signal
config : Config object
	Object containing WPD parameters
Returns
-------
list of numpy ndarray of size config.maxlevel
	WPD coefficients
"""
def wpd(signal, config):
	wp = pywt.WaveletPacket(
		signal,
		wavelet=config.wpd_wavelet,
		mode=config.mode,
		maxlevel=config.maxlevel)
	coefs = []
	for i in range(1, config.maxlevel+1):
		nodes = wp.get_level(i, order=config.order)
		value = np.array([rescale(n.data, config.samplerate) for n in nodes], 'd')
		coefs.append(value)
	return coefs


"""Returns CWT coefficients of wav signal
Parameters
----------
signal : numpy ndarray
	Input signal
config : Config object
	Object containing CWT parameters
Returns
-------
list of numpy ndarray of size config.maxlevel
	CWT coefficients
"""
def cwt(signal, config):
	coefs, freqs = pywt.cwt(
		data=signal,
		scales=config.scales,
		wavelet=config.cwt_wavelet,
		method=config.method)
	return coefs


def rescale(signal, rate):
	n = len(signal)
	return np.interp(np.linspace(0, n, rate), np.arange(n), signal)


def resample(signal, sr, n):
	for i in range(n):
		cut = np.random.randint(0, len(signal) - sr)
		yield signal[cut: cut + sr]


def __pad__(signal, sr):
	if len(signal) >= sr:
		return signal
	else:
		return np.pad(signal,
			pad_width=(sr - len(signal), 0),
			mode='constant',
			constant_values=(0, 0))


"""Returns MFCC features of wav signal
Parameters
----------
signal : numpy ndarray
	Input signal
config : Config object
	Object containing WPD parameters
Returns
-------
numpy ndarray of shape (signal_length/config.winlen, config.numcep)
	MFCC features
"""
def __mfcc__(signal, config):
	return mfcc(signal,
		samplerate=config.samplerate,
		winlen=config.winlen,
		winstep=config.winstep,
		numcep=config.numcep,
		nfilt=config.nfilt,
		nfft=config.nfft)


class Config:
	def __init__(self, samplerate, winlen, winstep, numcep, nfilt, nfft):
		self.samplerate = samplerate
		self.winlen = winlen
		self.winstep = winstep
		self.numcep = numcep
		self.nfilt = nfilt
		self.nfft = nfft
		self.lowfreq = 0
		self.highfreq = None
		self.preemph = 0.97
		self.ceplifter = 22
		self.appendEnergy = True
		self.wpd_wavelet = 'haar'
		self.cwt_wavelet = 'morl'
		self.mode = 'symmetric'
		self.order = 'freq'
		self.interpolation = 'nearest'
		self.method = 'fft'
		self.maxlevel = 4
		self.scales = np.array([2**x for x in range(self.maxlevel)])


"""
Builds MFCC feature dataset from wav signals dataset
Args:
	data_set ('acoustic', 'throat')
	split ('train', 'test')
	decomposition_strategy ('raw', 'wpd', 'cwt')
	decomposition_level (1, 2, 3, 4)
Example:
	To build MFCC feature vectors from level 2 WPD coefficients of acoustic training data
	python build_features.py acoustic train wpd 2
"""
if __name__ == '__main__':
	os.chdir('../../')
	DATASET = sys.argv[1]
	SPLIT = sys.argv[2]
	PATH = os.getcwd() + '/data/raw/'+DATASET+'/'+SPLIT+'/'

	config = Config(22050, 0.02, 0.01, 13, 26, 512)

	labels, filenames = grab_files(PATH)
	X_raw, X_wpd, X_cwt, y = [], [], [], []

	STRAT = sys.argv[3]
	if STRAT != 'raw':
		LEVEL = int(sys.argv[4])

	n = (200//6, 2300//6)[DATASET == 'acoustic']
	for label, fname in tqdm(zip(labels, filenames)):
		signal, rate = librosa.load(os.path.join(PATH, label, fname))
		signal = __pad__(signal, config.samplerate)
		if len(signal) > config.samplerate:
			samples = resample(signal, config.samplerate, n)
		else:
			samples = [signal]

		for sample in samples:
			if STRAT == 'raw':
				X_raw.append(__mfcc__(sample, config))
			elif STRAT == 'wpd':
				wpd_coefs = wpd(sample, config)
				features = []
				for level_coefs in wpd_coefs[LEVEL-1]:
					features.append(__mfcc__(level_coefs, config))
				X_wpd.append(features)
			elif STRAT == 'cwt':
				cwt_coefs = cwt(sample, config)
				X_cwt.append(__mfcc__(cwt_coefs[LEVEL-1], config))
			y.append(label)

	if STRAT == 'raw':
		X_raw = np.array(X_raw)
		pickle.dump(X_raw, open('data/processed/'+DATASET+'/'+SPLIT+'/X_raw.pickle', 'wb'), protocol=4)
	elif STRAT == 'wpd':
		X_wpd = np.array(X_wpd)
		X_wpd = np.reshape(X_wpd, (X_wpd.shape[0], X_wpd.shape[2], -1))
		pickle.dump(X_wpd, open('data/processed/'+DATASET+'/'+SPLIT+'/X_wpd_level'+str(LEVEL)+'.pickle', 'wb'), protocol=4)
	elif STRAT == 'cwt':
		X_cwt = np.array(X_cwt)
		pickle.dump(X_cwt, open('data/processed/'+DATASET+'/'+SPLIT+'/X_cwt_level'+str(LEVEL)+'.pickle', 'wb'), protocol=4)

	y = label_transform(y)
	label_index = y.columns.values
	y = y.values
	y = np.array(y)
	pickle.dump(y, open('data/processed/'+DATASET+'/'+SPLIT+'/y_'+SPLIT+'.pickle', 'wb'), protocol=4)

	del labels, filenames, X_raw, X_wpd, X_cwt
	gc.collect()
