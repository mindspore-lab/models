import os
import IPython.display as ipd
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler

import utils


def get_all_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list


AUDIO_DIR = 'fma_small/fma_small'
# Load metadata and features.
tracks = utils.load('fma_small/fma_metadata/tracks.csv')
genres = utils.load('fma_small/fma_metadata/genres.csv')
ipd.display(tracks['track'].head())
small = tracks[tracks['set', 'subset'] <= 'small']
# assert small.isin(tracks.index).all()
subset = tracks.index[tracks['set', 'subset'] <= 'small']

print(small.shape)
tracks = tracks.loc[subset]
train = tracks.index[tracks['set', 'split'] == 'training']
val = tracks.index[tracks['set', 'split'] == 'validation']
test = tracks.index[tracks['set', 'split'] == 'test']
print('{} training examples, {} validation examples, {} testing examples'.format(*map(len, [train, val, test])))

genres = list(LabelEncoder().fit(tracks['track', 'genre_top']).classes_)
#genres = list(tracks['track', 'genre_top'].unique())
print('Top genres ({}): {}'.format(len(genres), genres))
genres = list(MultiLabelBinarizer().fit(tracks['track', 'genres_all']).classes_)
print('All genres ({}): {}'.format(len(genres), genres))
labels_onehot = LabelBinarizer().fit_transform(tracks['track', 'genre_top'])
labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)
# labels_onehot.to_csv("test.csv")
directory = AUDIO_DIR
files = get_all_files(directory)
labels_onehot['path'] = files
train_file = labels_onehot.loc[train]

train_file.to_csv("train.csv",index=False)
val_file = labels_onehot.loc[val]
val_file.to_csv("val.csv",index=False)



