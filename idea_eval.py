import os.path
from tqdm import *
import pandas as pd

from get import load_data
from load import get_data
from process import node_embedding
from main_eval import joint_dev
from CONSTANTS import *
from Drain.demo import parse
import csv
import time
from utils.template_split import *
def model_eval(actual, pred):
    confusion = pd.crosstab(actual, pred, rownames=['Actual'], colnames=['Predicted'])
    try:
        TP = confusion.loc[1, 1]
    except:
        TP = 0
    try:
        TN = confusion.loc[0, 0]
    except:
        TN = 0
    try:
        FP = confusion.loc[0, 1]
    except:
        FP = 0
    try:
        FN = confusion.loc[1, 0]
    except:
        FN = 0

    print("TP={}, TN={}, FP={}, FN={}".format(TP, TN, FP, FN))

    out = {}
    out['ACR'] = round((TP + TN) / (TP + TN + FP + FN) * 100, 4)
    try:
        out['PRE'] = round(TP / (TP + FP) * 100, 4)
    except ZeroDivisionError:
        out['PRE'] = 0
    try:
        out['PF'] = round(FP / (FP + TN) * 100, 4)
    except ZeroDivisionError:
        out['PF'] = 0
    try:
        out['PD'] = round(TP / (TP + FN) * 100, 4)
    except ZeroDivisionError:
        out['PD'] = 0
    try:
        out['F1'] = round((2 * out['PRE'] * out['PD']) / (out['PRE'] + out['PD']), 4)
    except ZeroDivisionError:
        out['F1'] = 0
    print('Acc={}, PRE={}, Recall={}, F1={}'.format(out['ACR'], out['PRE'], out['PD'], out['F1']))
    # out['DI'] = round((TP / (TP + FN)) / (TN / (TN + FP)) * 100, 4)
    # out['TPR'] = round(TP / (TP + FN) * 100, 4)
    # out['TNR'] = round(TN / (FP + TN) * 100, 4)
    return out

def calculate_variance(data):
    n = len(data)
    if n == 0:
        return 0
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    return variance
if __name__ == '__main__':
    for file in ['hades']:
        J_VHL_ACC, J_VHL_PRE, J_VHL_PF, J_VHL_PD, J_VHL_F1 = [], [], [], [], []
        dataset = file.split('.')[0]
        times = []
        for iter in tqdm(range(1)):
            time1 = time.time()
            nodes, label, train_label, test_label, dev_label, nodes_test, len_t, len_tAd, u, id_train, \
             id_test, id_dev, seq_labels, seq_idxs, label_idxs = load_data(file, dataset)
            embeddings = node_embedding(nodes, 4)
            num_hg = len(embeddings)
            pred_jvhl = joint_dev(embeddings, label[:len_t], label[len_tAd:], label[len_t : len_tAd], dev_label, id_dev, u, len_tAd, lamda=1, mu=1, gama=5)

            y_jvhl = np.ones(len(test_label)) * -1
            y_vhl = np.zeros((num_hg, len(test_label)))
            for i in range(len(nodes_test)):
                y_jvhl[id_test[i]] = pred_jvhl[i]
            pred_labels = []
            idx_to_y = {idx: y for idx, y in zip(label_idxs, y_jvhl)}
            for i, idxs in enumerate(seq_idxs):
                pred_y = int(any(idx_to_y.get(idx, 0) == 1 for idx in idxs))
                pred_labels.append(pred_y)
            time2 = time.time()
            times.append(time2 - time1)
            print('-----MVHL-----')
            logisitic_reg = model_eval(seq_labels, pred_labels)
            J_VHL_ACC.append(logisitic_reg['ACR'])
            J_VHL_PRE.append(logisitic_reg['PRE'])
            J_VHL_PF.append(logisitic_reg['PF'])
            J_VHL_PD.append(logisitic_reg['PD'])
            J_VHL_F1.append(logisitic_reg['F1'])
        print(np.mean(times), np.std(times))
