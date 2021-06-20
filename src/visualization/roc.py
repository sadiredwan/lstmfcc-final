import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc, precision_score
import tensorflow as tf
import seaborn

if __name__ == '__main__':

	os.chdir('../../')
	model = load_model('models/wpd_mfcc_model.h5', compile=True)
	X_test = pickle.load(open('data/processed/acoustic/test/X_wpd_level1.pickle', 'rb'))
	y_test = pickle.load(open('data/processed/acoustic/test/y_test.pickle', 'rb'))
	y_pred = model.predict(X_test)
	ps_y_pred = [y.argmax() for y in y_pred]
	ps_y_test = [y.argmax() for y in y_test]
	ps = precision_score(ps_y_test, ps_y_pred, average='micro')

	n_classes = 12
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	labels = 'down go left no off on right silence stop unknown up yes'.split()

	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	f1 = plt.figure(figsize=(10, 8))
	for i in range(n_classes):
		plt.plot(fpr[i], tpr[i], label=labels[i] + ' vs. rest (score = %0.2f)' % roc_auc[i])
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic')
		plt.legend(loc="lower right")

	plt.show()

	f2 = plt.figure(figsize=(10, 8))
	matrix = tf.math.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
	matrix = matrix/matrix.numpy().sum(axis=1)[:, tf.newaxis]
	ax= plt.subplot()
	seaborn.heatmap(matrix, annot=True, ax=ax)
	ax.set_xlabel('Predicted labels')
	ax.set_ylabel('True labels')
	ax.set_title('Confusion Matrix')
	ax.xaxis.set_ticklabels(labels, rotation=45)
	ax.yaxis.set_ticklabels(labels, rotation=45)
	plt.show()
