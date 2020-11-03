import os

def get_labels_from_dir(dirName):
    listOfFile = os.listdir(dirName)
    labels_names = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            labels_names.append(entry)
    return labels_names