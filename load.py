import os

from tqdm import tqdm

from CONSTANTS import *
from Drain.demo import parse
import pandas as pd
import ast
import numpy as np
import re
def replace_nth(text, u, v, replacement):
    replacement = replacement.split(' ')
    # matches = list(re.finditer(r'<\*>', text))
    p = 0
    for i in range(v - u):
        matches = list(re.finditer(r'<\*>', text))
        # print(text, len(matches))
        match = matches[u]  # 第 n 个匹配
        start, end = match.span()
        text = text[:start] + replacement[p] + text[end:]
        p += 1
    return text

def get_data(file, dataset):
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'dataset', dataset)):
        # shutil.rmtree(os.path.join(PROJECT_ROOT, 'dataset', dataset))
        os.makedirs(os.path.join(PROJECT_ROOT, 'dataset', dataset))
        # continue
    # else:
    #     os.makedirs(os.path.join(PROJECT_ROOT, 'dataset', dataset))
    # train_file = os.path.join(PROJECT_ROOT, 'dataset', dataset, dataset + '_train.log')
    # test_file = os.path.join(PROJECT_ROOT, 'dataset', dataset, dataset + '_test.log')
    if not os.path.isfile(os.path.join(PROJECT_ROOT, os.pardir, 'dataset', dataset, dataset + '.log_structured.csv')):
        parse(file, dataset, '', os.path.join(PROJECT_ROOT, os.pardir, 'dataset', dataset))
    data = pd.read_csv(os.path.join(PROJECT_ROOT, os.pardir, 'dataset', dataset, dataset + '.log_structured.csv'))
    normal_samples = data[data['Label'] == '-']
    anomaly_samples = data[data['Label'] != '-']
    normal_samples = normal_samples.sample(frac=1).reset_index(drop=True)
    anomaly_samples = anomaly_samples.sample(frac=1).reset_index(drop=True)
    len_n_train = int(0.6 * len(normal_samples))
    len_a_train = int(0.6 * len(anomaly_samples))
    len_n_dev = int(0.1 * len(normal_samples))
    len_a_dev = int(0.1 * len(anomaly_samples))
    train_logs = pd.concat([normal_samples[:len_n_train], anomaly_samples[:len_a_train]])
    dev_logs = pd.concat(([normal_samples[len_n_train : len_n_train + len_n_dev], anomaly_samples[len_a_train : len_a_train + len_a_dev]]))
    test_logs = pd.concat([normal_samples[len_n_train + len_n_dev:], anomaly_samples[len_a_train + len_a_dev:]])
    test_logs_labels = np.concatenate([np.zeros(len(normal_samples) - (len_n_train + len_n_dev)), np.ones(len(anomaly_samples) - (len_a_train + len_a_dev))])
    label_idxs = np.array(test_logs['LineId'])
    train_logs_labels = np.concatenate([np.zeros(len_n_train), np.ones(len_a_train)])
    dev_logs_labels = np.concatenate([np.zeros(len_n_dev), np.ones(len_a_dev)])
    train_nodes = train_logs['EventTemplate'].unique()
    test_nodes = test_logs['EventTemplate'].unique()
    dev_nodes = dev_logs['EventTemplate'].unique()
    train_nodes = np.array(train_nodes)
    test_nodes = np.array(test_nodes)
    dev_nodes = np.array(dev_nodes)
    replace = {}
    extra_labels = []
    extra_nodes = []
    for i, node in enumerate(train_nodes):
        l_ = train_logs[train_logs['EventTemplate'] == node]['Label']
        y_ = 0
        for _ in l_:
            y_ += 0 if _ == '-' else 1
        if y_ == 0 or y_ == len(l_):
            extra_labels.append(int(1 if y_ > 0 else 0))
            extra_nodes.append(node)
        if y_ > 0 and len(l_) != y_:
            para_n = train_logs[(train_logs['EventTemplate'] == node) & (train_logs['Label'] == '-')][
                'ParameterList'].drop_duplicates()
            para_a = train_logs[(train_logs['EventTemplate'] == node) & (train_logs['Label'] != '-')][
                'ParameterList'].drop_duplicates()
            # replace[node] = True
            para_n = list(para_n)
            para_a = list(para_a)
            a, b = 0, 0
            while a < len(para_n) and len(ast.literal_eval(para_n[a])) == 0:
                a += 1
            while b < len(para_a) and len(ast.literal_eval(para_a[b])) == 0:
                b += 1
            if a == len(para_n) or b == len(para_a):
                level_n = train_logs[(train_logs['EventTemplate'] == node) & (train_logs['Label'] == '-')][
                    'Level'].drop_duplicates()
                level_a = train_logs[(train_logs['EventTemplate'] == node) & (train_logs['Label'] != '-')][
                    'Level'].drop_duplicates()
                level_n = list(level_n)
                level_a = list(level_a)
                flag = 1
                for level in level_a:
                    if level in level_n:
                        flag = 0
                        break
                if flag == 1:
                    # print('normal')
                    for j, level in enumerate(level_n):
                        node_ = node + ' ' + level
                        indexer = (train_logs['EventTemplate'] == node) & (train_logs['Label'] == '-') & (
                                train_logs['Level'] == level)
                        train_logs.loc[indexer, 'EventTemplate'] = node_
                        if node_ not in train_nodes:
                            extra_nodes.append(node_)
                            extra_labels.append(int(0))
                    # print('abnormal')
                    for level in level_a:
                        node_ = node + ' ' + level
                        indexer = (train_logs['EventTemplate'] == node) & (train_logs['Label'] != '-') & (
                                train_logs['Level'] == level)
                        train_logs.loc[indexer, 'EventTemplate'] = node_
                        if node_ not in train_nodes:
                            extra_nodes.append(node_)
                            extra_labels.append(int(1))
                replace[node] = -1
                # train_labels[i] = int(1)
                continue

            n_, a_ = ast.literal_eval(para_n[a]), ast.literal_eval(para_a[b])
            k = 0
            while (k < len(n_) and k < len(a_) and (n_[len(n_) - k - 1] == a_[len(a_) - k - 1])):
                k += 1
            replace[node] = k
            for j, para in enumerate(para_n):
                actual_list = ast.literal_eval(para)
                node_ = node
                u_ = 0
                for jj in range(len(actual_list) - k - 1):
                    u_ += len(actual_list[jj].split(' '))
                v_ = u_ + len(actual_list[len(actual_list) - k - 1].split(' '))
                rep = actual_list[len(actual_list) - k - 1] if k < len(actual_list) else '<*>'
                node_ = replace_nth(node_, u_, v_, rep)
                indexer = (train_logs['EventTemplate'] == node) & (train_logs['Label'] == '-') & (
                        train_logs['ParameterList'] == para)
                train_logs.loc[indexer, 'EventTemplate'] = node_
                if node_ not in train_nodes:
                    extra_nodes.append(node_)
                    extra_labels.append(int(0))
            for para in para_a:
                actual_list = ast.literal_eval(para)
                node_ = node
                u_ = 0
                for jj in range(len(actual_list) - k - 1):
                    u_ += len(actual_list[jj].split(' '))
                v_ = u_ + len(actual_list[len(actual_list) - k - 1].split(' '))
                rep = actual_list[len(actual_list) - k - 1] if k < len(actual_list) else '<*>'
                node_ = replace_nth(node_, u_, v_, rep)
                indexer = (train_logs['EventTemplate'] == node) & (train_logs['Label'] != '-') & (
                        train_logs['ParameterList'] == para)
                train_logs.loc[indexer, 'EventTemplate'] = node_
                if node_ not in train_nodes:
                    extra_nodes.append(node_)
                    extra_labels.append(int(1))
    extra_nodes_ = []
    extra_labels_ = []
    for i, extra_node in enumerate(extra_nodes):
        if extra_node not in extra_nodes_:
            extra_nodes_.append(extra_node)
            extra_labels_.append(extra_labels[i])
    train_nodes = np.array(extra_nodes_)
    train_labels = np.array(extra_labels_, dtype=int)

    extra_nodes = []
    extra_labels = []
    for i, node in enumerate(dev_nodes):
        l_ = dev_logs[dev_logs['EventTemplate'] == node]['Label']
        y_ = 0
        for _ in l_:
            y_ += 0 if _ == '-' else 1
        if node not in replace.keys():
            extra_labels.append(int(1 if y_ > 0 else 0))
            extra_nodes.append(node)
        else:
            # print(node)
            para_ = dev_logs[dev_logs['EventTemplate'] == node]['ParameterList'].drop_duplicates()
            para_ = list(para_)
            if replace[node] == -1:
                level_ = dev_logs[dev_logs['EventTemplate'] == node]['Level'].drop_duplicates()
                level_ = list(level_)
                for j, level in enumerate(level_):
                    node_ = node + ' ' + level
                    indexer = (dev_logs['EventTemplate'] == node) & (dev_logs['Level'] == level)
                    _y = list(dev_logs[(dev_logs['EventTemplate'] == node) & (dev_logs['Level'] == level)][
                                  'Label'].drop_duplicates())
                    dev_logs.loc[indexer, 'EventTemplate'] = node_
                    # print(_y)
                    _ = 0
                    if len(_y) == 1:
                        _ = 1 if '-' not in _y else 0
                    else:
                        _ = 1
                    if node_ not in dev_nodes:
                        extra_nodes.append(node_)
                        extra_labels.append(int(_))
                continue
            k = replace[node]
            # print(node)
            for j, para in enumerate(para_):
                actual_list = ast.literal_eval(para)
                rep = actual_list[len(actual_list) - k - 1] if k < len(actual_list) else '<*>'
                node_ = node
                u_ = 0
                for jj in range(len(actual_list) - k - 1):
                    u_ += len(actual_list[jj].split(' '))
                v_ = u_ + len(actual_list[len(actual_list) - k - 1].split(' '))
                rep = actual_list[len(actual_list) - k - 1] if k < len(actual_list) else '<*>'
                node_ = replace_nth(node_, u_, v_, rep)
                _y = list(dev_logs[(dev_logs['EventTemplate'] == node) & (dev_logs['ParameterList'] == para)][
                              'Label'].drop_duplicates())
                indexer = (dev_logs['EventTemplate'] == node) & (dev_logs['ParameterList'] == para)
                dev_logs.loc[indexer, 'EventTemplate'] = node_
                # print(_y)
                _ = 0
                if len(_y) == 1:
                    _ = 1 if '-' not in _y else 0
                else:
                    _ = 1
                if node_ not in dev_nodes:
                    extra_nodes.append(node_)
                    extra_labels.append(int(_))
                # print(node_)
    extra_nodes_ = []
    extra_labels_ = []
    for i, extra_node in enumerate(extra_nodes):
        if extra_node not in extra_nodes_:
            extra_nodes_.append(extra_node)
            extra_labels_.append(extra_labels[i])

    dev_nodes = np.array(extra_nodes_)
    dev_labels = np.array(extra_labels_, dtype=int)

    extra_nodes = []
    extra_labels = []
    for i, node in enumerate(test_nodes):
        l_ = test_logs[test_logs['EventTemplate'] == node]['Label']
        y_ = 0
        for _ in l_:
            y_ += 0 if _ == '-' else 1
        if node not in replace.keys():
            extra_labels.append(int(1 if y_ > 0 else 0))
            extra_nodes.append(node)
        else:
            # print(node)
            para_ = test_logs[test_logs['EventTemplate'] == node]['ParameterList'].drop_duplicates()
            para_ = list(para_)
            if replace[node] == -1:
                level_ = test_logs[test_logs['EventTemplate'] == node]['Level'].drop_duplicates()
                level_ = list(level_)
                for j, level in enumerate(level_):
                    node_ = node + ' ' + level
                    indexer = (test_logs['EventTemplate'] == node) & (test_logs['Level'] == level)
                    _y = list(test_logs[(test_logs['EventTemplate'] == node) & (test_logs['Level'] == level)][
                                  'Label'].drop_duplicates())
                    test_logs.loc[indexer, 'EventTemplate'] = node_
                    # print(_y)
                    _ = 0
                    if len(_y) == 1:
                        _ = 1 if '-' not in _y else 0
                    else:
                        _ = 1
                    # print('normal' if int(_) == 0 else 'abnormal', node_)
                    # if j == 0:
                    #     test_logs[i] = node_
                    #     test_logs[i] = int(_)
                    # else:
                    if node_ not in test_nodes:
                        extra_nodes.append(node_)
                        extra_labels.append(int(_))
                # test_labels[i] = int(1)
                continue
            k = replace[node]
            # print(node)
            for j, para in enumerate(para_):
                actual_list = ast.literal_eval(para)
                u_ = 0
                node_ = node
                for jj in range(len(actual_list) - k - 1):
                    u_ += len(actual_list[jj].split(' '))
                v_ = u_ + len(actual_list[len(actual_list) - k - 1].split(' '))
                rep = actual_list[len(actual_list) - k - 1] if k < len(actual_list) else '<*>'
                node_ = replace_nth(node_, u_, v_, rep)
                _y = list(test_logs[(test_logs['EventTemplate'] == node) & (test_logs['ParameterList'] == para)][
                              'Label'].drop_duplicates())
                indexer = (test_logs['EventTemplate'] == node) & (test_logs['ParameterList'] == para)
                test_logs.loc[indexer, 'EventTemplate'] = node_
                # print(_y)
                _ = 0
                if len(_y) == 1:
                    _ = 1 if '-' not in _y else 0
                else:
                    _ = 1
                # print('normal' if int(_) == 0 else 'abnormal', node_)
                # if j == 0:
                #     test_logs[i] = int(_)
                #     test_nodes[i] = node_
                # else:
                if node_ not in test_nodes:
                    extra_nodes.append(node_)
                    extra_labels.append(int(_))
                # print(node_)
    extra_nodes_ = []
    extra_labels_ = []
    for i, extra_node in enumerate(extra_nodes):
        if extra_node not in extra_nodes_:
            extra_nodes_.append(extra_node)
            extra_labels_.append(extra_labels[i])

    test_nodes = np.array(extra_nodes_)
    test_labels = np.array(extra_labels_, dtype=int)
    id_train = {}
    for i, node in enumerate(train_nodes):
        id_train[i] = np.where(train_logs['EventTemplate'] == node)[0]
    id_dev = {}
    for i, node in enumerate(dev_nodes):
        id_dev[i] = np.where(dev_logs['EventTemplate'] == node)[0]
    id_test = {}
    for i, node in enumerate(test_nodes):
        id_test[i] = np.where(test_logs['EventTemplate'] == node)[0]
    seq_labels = []
    seq_idxs = []
    if dataset == 'hades':
        pbar = tqdm(total=len(test_logs))
        l_ = test_logs['Label'].to_numpy()
        lineId = test_logs['LineId'].to_numpy()
        for i in range(0, len(test_logs), 120):
            l_chunk = l_[i:i + 120]
            id_chunk = lineId[i:i + 120]

            seq_label = int(np.any(l_chunk != '-'))  # 利用 NumPy 加速 any 判断

            seq_labels.append(seq_label)
            seq_idxs.append(id_chunk)
            pbar.update(len(l_chunk))
    else:
        _nodes = test_logs['Node'].unique()
        _nodes = np.array(_nodes)
        pbar = tqdm(total=len(_nodes))
        grouped = test_logs.groupby('Node')

        for _n, group in grouped:
            l_ = group['Label'].to_numpy()
            lineId = group['LineId'].to_numpy()

            for i in range(0, len(lineId), 120):
                l_chunk = l_[i:i + 120]
                id_chunk = lineId[i:i + 120]

                seq_label = int(np.any(l_chunk != '-'))  # 利用 NumPy 加速 any 判断

                seq_labels.append(seq_label)
                seq_idxs.append(id_chunk)
            pbar.update(1)
        seq_labels = np.array(seq_labels)
    u = np.zeros(len(train_nodes))
    for i in range(len(train_nodes)):
        u[i] = len(id_train[i])
    sumn = np.sum(u[train_labels == 0])
    suma = np.sum(u[train_labels == 1])
    u[train_labels == 0] /= sumn
    u[train_labels == 1] /= suma
    # for node in test_nodes:
    #     if node not in train_nodes:
    #         print(node)
    return np.concatenate([train_nodes, dev_nodes, test_nodes]), \
        np.concatenate([train_labels, dev_labels, test_labels]), \
        train_logs_labels, test_logs_labels, dev_logs_labels, test_nodes, \
        len(train_nodes), len(train_nodes) + len(dev_nodes), u, id_train, id_test, id_dev, \
        seq_labels, seq_idxs, label_idxs