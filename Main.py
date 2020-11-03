
import os
import subprocess

extr = 0  # 1 to extract the features, 0 don't extract
mode = 0  # 1 to train the network, 0 to use a pre-trained model
prj_dir = os.getcwd()

subprocess.call(["python", prj_dir+r"\\lib\\main.py",
                 "-e", str(extr), '-m', str(mode), '--prj_dir', prj_dir])
