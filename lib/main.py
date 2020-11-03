import argparse
import os.path
import urllib.request
import tarfile
import shutil
import numpy as np
import get_feat as get_feat
import get_labels as get_labels
import cnn as cnn
import conf_matrix as conf_mat





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", type=int, default=1,
                        help="1 to extract features, 0 don't extract")
    parser.add_argument("-m", type=int, default=1,
                        help="1 to train, 0 to use saved model")
    parser.add_argument("--prj_dir",
                        help="working directory")
    args = parser.parse_args()

    extr = args.e
    prj_dir = args.prj_dir
    if args.m == 1:
        mode = 'train'
    elif args.m == 0:
        mode = 'test'

    n_labels = 10  # 10 genres in database
    
    data_dir = prj_dir + r'\data\raw'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        url = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
        print("download start!")
        filename, headers = urllib.request.urlretrieve(url, filename=data_dir+r'\genres.tar.gz')
        print("download complete!")
        print('extracting files from genres.tar.gz!')
        tar = tarfile.open(data_dir+r'\genres.tar.gz', "r")
        tar.extractall(data_dir)
        tar.close()
        print('Done!')

    # get labels from directories:
    classes = get_labels.get_labels_from_dir(data_dir+r'\genres')

    # extract features (if extr == 1):
    feat_dir = prj_dir + r'\data\feat'
    if extr == 1:
        if os.path.exists(feat_dir):
            shutil.rmtree(feat_dir)
        os.makedirs(feat_dir)
        print('extracting features!')
        get_feat.feat_extract(data_dir, feat_dir, classes)
        print('Done!')

    # define CNN and train it:
    n_classes = n_labels
    im_size = (204, 204)
    pred, labels = cnn.main(mode, n_classes, im_size, prj_dir, feat_dir, classes, batch_size=50)

    # confusion matrix:
    print('Get confusion matrix')
    conf_mat.main(labels, np.argmax(pred, axis=-1), classes, prj_dir)

    print('Done!')
