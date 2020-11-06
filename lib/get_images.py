import os
from matplotlib.pyplot import imread
import random
import numpy as np

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
        elif fullPath.endswith('.png'):
            allFiles.append(fullPath)
    return allFiles

def main(dir):
    all_files = getListOfFiles(dir)
    random.shuffle(all_files)
    n = len(all_files)
    data = []
    # sub sample 1/3 of the database
    for i in range(n//3):
        im = imread(all_files[i])[:,:,:3]
        data.append(im)
    return data