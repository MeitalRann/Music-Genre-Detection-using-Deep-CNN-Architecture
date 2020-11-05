import os
from collections import Counter
import random
import numpy as np

# set seed:
seed_value = 1
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

def group(pred, labels, test_names):
    # group together labels and predictions from the same music recording
    n = len(test_names)
    pred_grouped = {}
    labels_grouped = []
    prev = [int(s) for s in test_names[0].replace('.','_').split('_') if s.isdigit()][0]
    ind = 0
    for i in range(n):
        num_file = [int(s) for s in test_names[i].replace('.','_').split('_') if s.isdigit()][0]
        if num_file == prev:
            try:
                pred_grouped[ind].append(pred[i])
            except:
                pred_grouped[ind] = [pred[i]]
                labels_grouped.append(labels[i])
        else:
            ind+=1
        prev = num_file
    return pred_grouped, labels_grouped


def vote(pred):
    n = len(pred)
    majority = []
    for i in range(n):
        c = Counter(pred[i])
        vals = c.most_common()
        max_c = vals[0][1]
        m = [vals[j][0] for j in range(len(vals)) if vals[j][1]==max_c]
        majority.append(random.choice(m))
    return majority


def main(pred, labels, test_names):
    pred_grouped, labels_grouped = group(pred, labels, test_names)
    pred_majority = vote(pred_grouped)
    return pred_majority, labels_grouped