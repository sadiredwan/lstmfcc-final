import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow_datasets as tfds


def download_tfrecord():
    os.chdir('../../data/external')
    data_dir = os.getcwd()
    tfds.load('speech_commands',
        data_dir=data_dir,
        split=['test', 'train', 'validation'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True)


def download_raw():
    os.chdir('../../data/raw')
    data_dir = os.getcwd()
    dm = tfds.download.DownloadManager(download_dir=data_dir)
    train = dm.download_and_extract('http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz')
    test = dm.download_and_extract('http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz')


if __name__ == '__main__':
    download_raw()
