import math

import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy import sparse

from hyperG import HyperG

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
    return out
def cal_DF(i, F, Fi):
    ret = 0
    pi = np.argmax(Fi, axis=1)
    for j in range(len(F)):
        ret += (1 if j != i else 0) * np.sum(np.array(pi - np.argmax(F[j], axis=1)) ** 2)
    return ret
def cal_F_V(i, F, alpha):
    F_ret = np.zeros((F[i].shape[0], F[i].shape[1]))
    for j in range(len(F)):
        F_ret += (1 if j != i else 0) * F[j]
    return F_ret

def joint_dev(embedding, y_, test_y, dev_y, dev_label, id_dev, u, len_dev, lamda=1, mu=1, gama=1):
    _y = np.ones(dev_y.shape[0] + test_y.shape[0]) * -1
    hg = [HyperG(emb, y_, u=u, K=4) for emb in embedding]

    F = [hg_.predict() for hg_ in hg]
    Y_dev = np.zeros((dev_y.shape[0], 2))
    for i in range(len(dev_y)):
        Y_dev[i, :] = 0
        Y_dev[i, dev_y[i]] = 1
    res = []
    nt = F[0].shape[0]
    I = sparse.eye(nt)
    Y = hg[0].Y
    num_hg = len(hg)
    alpha = np.ones(num_hg) * (1. / num_hg)
    L2 = [I, I, I]
    pred_to_logs = []
    phi = np.zeros(num_hg)
    for i in range(len(hg)):
        row_sums = F[i].sum(axis=1)
        row_sums = row_sums[:, np.newaxis]
        F[i] /= row_sums
        L2[i] = hg[i].L2
        pred = np.argmax(F[i], axis=1)
        pred = pred[len(y_):len_dev]
        p_t_l = np.zeros(dev_label.shape[0])
        for j in range(len(pred)):
            p_t_l[id_dev[j]] = pred[j]
        pred_to_logs.append(p_t_l)
        phi[i] = lamda * np.sum(np.array(F[i][len(y_): len(y_) + dev_y.shape[0], :] - Y_dev) ** 2) \
                 + gama * np.sum(np.array(pred_to_logs[i] - dev_label) ** 2)
    loss = -1
    iters = 10
    base = 1 if (num_hg * np.max(phi) - np.sum(phi)) / 2 == 0 else (num_hg * np.max(phi) - np.sum(phi)) / 2
    zeta = np.power(10.0, int(math.log10(base) + 1))
    b = [int(math.log10(base) + 1)]
    for i in range(num_hg):
        alpha[i] = 1 / num_hg + (np.sum(phi) / (2 * num_hg * zeta) - (phi[i] / (2 * zeta)))
    print('alpha=', alpha)
    print('--MVHL--')
    Last_F = F
    Last_alpha = alpha
    for iter in range(iters):
        F = Last_F
        pred_to_logs = []
        phi = np.zeros(num_hg)
        for i in range(num_hg):
            F[i] = inv(alpha[i] * (L2[i].toarray()) + lamda * alpha[i] * I + mu * I). \
                dot(lamda * alpha[i] * Y + mu * cal_F_V(i, Last_F, alpha))
            row_sums = F[i].sum(axis=1)
            F[i] /= row_sums
            pred = np.argmax(F[i], axis=1)
            pred = pred[len(y_):len_dev]
            p_t_l = np.zeros(dev_label.shape[0])
            for j in range(len(pred)):
                p_t_l[id_dev[j]] = pred[j]
            pred_to_logs.append(p_t_l)
            phi[i] = lamda * np.sum(np.array(F[i][len(y_): len(y_) + dev_y.shape[0], :] - Y_dev) ** 2) \
                     + gama * np.sum(np.array(pred_to_logs[i] - dev_label) ** 2)

        base = 1 if (num_hg * np.max(phi) - np.sum(phi)) / 2 == 0 else (num_hg * np.max(phi) - np.sum(phi)) / 2
        b.append(int(math.log10(base) + 1))
        zeta = np.power(10.0, int(math.log10(base) + 1))
        for i in range(num_hg):
            alpha[i] = 1 / num_hg + (np.sum(phi) / (2 * num_hg * zeta) - (phi[i] / (2 * zeta)))
        print('alpha=', alpha)
        _loss = zeta * np.sum(alpha ** 2)
        for i in range(num_hg):
            _loss += mu * cal_DF(i, F, F[i]) + lamda * alpha[i] * np.sum(
                np.array(F[i][len_dev - dev_y.shape[0] : len_dev, :] - Y_dev) ** 2) \
                     + gama * alpha[i] * np.sum(np.array(dev_label - np.array(pred_to_logs[i])) ** 2)

        F_ret = F[0] * alpha[0]
        for i in range(1, num_hg):
            F_ret += F[i] * alpha[i]
        pre = np.argmax(np.array(F_ret), axis=1)[len_dev:]
        reg = model_eval(pre, test_y)
        print('iter: {}, Accuracy={}, Precision={}, Recall = {}, F1={}'.format(iter, reg['ACR'], reg['PRE'], reg['PD'], reg['F1']))
        if loss == -1 or _loss < loss:
            loss = _loss
            Last_F = F
            Last_alpha = alpha
        else:
            break
    F = Last_F
    alpha = Last_alpha
    F_ret = F[0] * alpha[0]
    for i in range(1, num_hg):
        F_ret += F[i] * alpha[i]
    pred = np.argmax(np.array(F_ret), axis=1)[len_dev:]
    return pred