import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import librosa.display
from tqdm.auto import tqdm
import random

# set seed:
seed_value = 1
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        elif fullPath.endswith('.wav'):
            allFiles.append(fullPath)

    return allFiles

def log_spec(data, n_fft=2048, hop_length=512, n_mels=128, m = 1, ):
    if m == 0:
        # Calculate the spectrogram as the square of the complex magnitude of the STFT
        spectrogram = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann')) ** 2
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        out = spectrogram_db
    else:
        # calculate the mel-spectrogram
        melspectrogram = librosa.feature.melspectrogram(data, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
        out = melspectrogram
    return out

def feat_extract(data_dir, out_dir, labels_names, size=(204,204)):
    # create train, test and valid dirs:
    train_dir = out_dir + r'\\train'
    test_dir = out_dir + r'\\test'
    valid_dir = out_dir + r'\\valid'
    os.makedirs(train_dir)
    os.makedirs(test_dir)
    os.makedirs(valid_dir)
    # create genre folders:
    for j in range(len(labels_names)):
        dir = train_dir + r'\\' + labels_names[j]
        os.makedirs(dir)
    for j in range(len(labels_names)):
        dir = test_dir + r'\\' + labels_names[j]
        os.makedirs(dir)
    for j in range(len(labels_names)):
        dir = valid_dir + r'\\' + labels_names[j]
        os.makedirs(dir)

    n_fft = 2048  # Size of the FFT, which will also be used as the window length
    hop_length = 512  # Step or stride between windows.
    n = size[0]
    m = size[1]

    all_files = getListOfFiles(data_dir)
    random.shuffle(all_files)
    n_files = len(all_files)
    train_lim = round(0.7*n_files)  # number of files to place in train (70%)
    valid_lim = train_lim + round(0.2*n_files) # number of files to place in valid (20%)

    test_files = []
    for i in tqdm(range(len(all_files))):
        # Load sample audio file
        file_i = all_files[i]
        label = file_i.split('\\')[-1].split('.')[0]
        data, sr = librosa.load(file_i, mono=True)
        data = np.array(data).astype(float)
        # extract feature:
        feat = log_spec(data, n_fft, hop_length, n_mels=m)

        # divide each spectrogram to 204 sec images with 50% overlap
        n_images = len(feat[1,:])//(n//2)-1  # find how many images can be made from each recording
        for j in range(n_images):
            sub_feat = feat[:, j*(n//2):j*(n//2)+n]
            # save feature in right directory
            if i <= train_lim:
                name = train_dir + r'\\' + label + r'\\spectrogram_' + str(i) + '_' + str(j) + '.png'
            elif i > train_lim and i <= valid_lim:
                name = valid_dir + r'\\' + label + r'\\spectrogram_' + str(i) + '_' + str(j) + '.png'
            else:
                name = test_dir + r'\\' + label + r'\\spectrogram_' + str(i) + '_' + str(j) + '.png'
                test_files.append(name)
            plt.imsave(name, sub_feat)
        # librosa.display.specshow(feat, sr=sr, y_axis='log', x_axis='time', hop_length=hop_length)
        # plt.title('Power spectrogram')
        # plt.colorbar(format='%+2.0f dB')
        # plt.tight_layout()
        # plt.show()