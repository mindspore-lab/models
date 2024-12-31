import gzip
import h5py  # 新增
import os
import numpy as np
import logging
from tqdm import tqdm
from utils import prepare_words_list
from src.model_utils.config import config, prepare_model_settings


class AudioProcessor():
    """Handles loading, partitioning, and preparing audio training data."""

    def __init__(self, data_url, data_dir, silence_percentage, unknown_percentage,
                 wanted_words, validation_percentage, testing_percentage,
                 model_settings):
        self.data_dir = data_dir
        self.maybe_download_and_extract_dataset(data_url, data_dir)
        self.prepare_data_index(silence_percentage, unknown_percentage,
                                wanted_words, validation_percentage,
                                testing_percentage)
        self.prepare_data(model_settings)

    def maybe_download_and_extract_dataset(self, data_url, dest_directory):
        '''Download and extract .h5.gz dataset'''
        if not data_url:
            return
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)

        # Download the file if it does not exist
        if not os.path.exists(filepath):
            try:
                filepath, _ = urllib.request.urlretrieve(data_url, filepath)
            except:
                logging.error('Failed to download URL: %s to folder: %s', data_url, filepath)
                raise
            print('Download complete.')
        # Decompress .h5.gz file
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rb') as f_in:
                h5_path = filepath[:-3]  # Remove .gz extension
                with open(h5_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            logging.info('Successfully extracted %s', h5_path)

    def prepare_data_index(self, silence_percentage, unknown_percentage,
                           wanted_words, validation_percentage,
                           testing_percentage):
        '''Prepare data index from .h5 file'''
        h5_filepath = os.path.join(self.data_dir, 'dataset.h5')
        if not os.path.exists(h5_filepath):
            raise Exception('No .h5 file found at ' + h5_filepath)

        # Read the HDF5 dataset
        with h5py.File(h5_filepath, 'r') as h5_file:
            # Assuming dataset is organized into groups for 'training', 'validation', and 'testing'
            self.data_index = {
                'training': h5_file['training_data'][:],
                # 'validation': h5_file['validation_data'][:],
                # 'testing': h5_file['testing_data'][:]
            }
            self.labels = {
                'training': h5_file['training_labels'][:],
                # 'validation': h5_file['validation_labels'][:],
                # 'testing': h5_file['testing_labels'][:]
            }

        # Prepare words list and word-to-index mapping
        self.words_list = prepare_words_list(wanted_words)
        self.word_to_index = {word: idx for idx, word in enumerate(self.words_list)}

    def prepare_data(self, model_settings):
        '''Prepare data from the loaded HDF5 dataset'''
        if not os.path.exists(config.download_feat_dir):
            os.makedirs(config.download_feat_dir, exist_ok=True)

        for mode in ['training', 'validation', 'testing']:
            data = self.data_index[mode]
            labels = self.labels[mode]

            # Prepare feature extraction and data reshaping
            sample_count = len(data)
            extracted_data = np.zeros((sample_count, model_settings['fingerprint_size']))
            for i in tqdm(range(sample_count)):
                feature = self.extract_features(data[i], model_settings)
                extracted_data[i, :] = feature

            # Save the processed data and labels
            np.save(os.path.join(config.download_feat_dir, f'{mode}_data.npy'), extracted_data)
            np.save(os.path.join(config.download_feat_dir, f'{mode}_label.npy'), labels)

    def extract_features(self, wav_data, model_settings):
        '''Feature extraction process, replacing MFCC extraction'''
        # Your feature extraction logic goes here. In case you still want to use MFCC:
        feature = mfcc(wav_data, samplerate=config.sample_rate, winlen=config.window_size_ms / 1000,
                       winstep=config.window_stride_ms / 1000,
                       numcep=config.dct_coefficient_count, nfilt=40, nfft=1024, lowfreq=20,
                       highfreq=7000).flatten()
        return feature


if __name__ == '__main__':
    print('Start processing .h5.gz data')
    model_settings_1 = prepare_model_settings(
        len(prepare_words_list(config.wanted_words.split(','))),
        config.sample_rate, config.clip_duration_ms, config.window_size_ms,
        config.window_stride_ms, config.dct_coefficient_count)
    audio_processor = AudioProcessor(
        config.download_data_url, config.data_dir, config.silence_percentage,
        config.unknown_percentage,
        config.wanted_words.split(','), config.validation_percentage,
        config.testing_percentage, model_settings_1)
