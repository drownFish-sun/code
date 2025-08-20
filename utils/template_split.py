import re
import ast
import numpy as np
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
def get_label_count(logs, node):
    """统计该模板为异常的数量"""
    labels = logs[logs['EventTemplate'] == node]['Label']
    return sum(1 if lbl != '-' else 0 for lbl in labels), len(labels)

def parse_param_list(param_list):
    """处理字符串形式的参数列表为真实的 Python list"""
    for item in param_list:
        parsed = ast.literal_eval(item)
        if parsed:
            return parsed
    return []

def should_split_by_level(levels_normal, levels_abnormal):
    """判断是否应通过 Level 拆分"""
    return not any(level in levels_normal for level in levels_abnormal)

def split_by_level(logs, node, label, target_nodes, target_labels, existing_nodes):
    """根据 Level 拆分节点"""
    levels = logs[logs['EventTemplate'] == node]['Level'].drop_duplicates()
    for level in levels:
        node_new = f"{node} {level}"
        indexer = (logs['EventTemplate'] == node) & (logs['Level'] == level)
        logs.loc[indexer, 'EventTemplate'] = node_new
        label_vals = logs.loc[indexer, 'Label'].drop_duplicates().tolist()
        lbl = 1 if any(l != '-' for l in label_vals) else 0
        if node_new not in existing_nodes:
            target_nodes.append(node_new)
            target_labels.append(lbl)

def replace_by_param(logs, node, param_normal, param_abnormal, target_nodes, target_labels, label_val):
    """使用参数尾部差异对模板进行替换"""
    p_n = parse_param_list(param_normal)
    p_a = parse_param_list(param_abnormal)
    if not p_n or not p_a:
        return False, 0 # 无法进行替换

    # 找公共后缀参数个数
    k = 0
    while k < len(p_n) and k < len(p_a) and p_n[-k - 1] == p_a[-k - 1]:
        k += 1

    def update_template(row):
        params = ast.literal_eval(row['ParameterList'])
        if k >= len(params): return row['EventTemplate']
        u = sum(len(p.split(' ')) for p in params[:-k-1])
        v = u + len(params[-k - 1].split(' '))
        return replace_nth(row['EventTemplate'], u, v, params[-k - 1])

    mask = logs['EventTemplate'] == node
    logs.loc[mask, 'EventTemplate'] = logs[mask].apply(update_template, axis=1)

    updated_nodes = logs[mask]['EventTemplate'].unique()
    for n in updated_nodes:
        if n not in target_nodes:
            target_nodes.append(n)
            target_labels.append(label_val)
    return True, k

def process_logs(logs, nodes, replace_map=None):
    """核心函数，对 logs 进行处理"""
    if replace_map is None:
        replace_map = {}

    new_nodes = []
    new_labels = []

    for node in nodes:
        y_abnormal, total = get_label_count(logs, node)

        if y_abnormal == 0 or y_abnormal == total:
            new_nodes.append(node)
            new_labels.append(1 if y_abnormal > 0 else 0)
            continue

        if replace_map is not None and node in replace_map:
            # 已记录，使用 map 中的信息
            if replace_map[node] == -1:
                split_by_level(logs, node, 'Label', new_nodes, new_labels, nodes)
            else:
                k = replace_map[node]
                _, _ = replace_by_param(logs, node,
                                 logs[(logs['EventTemplate'] == node) & (logs['Label'] == '-')]['ParameterList'].drop_duplicates(),
                                 logs[(logs['EventTemplate'] == node) & (logs['Label'] != '-')]['ParameterList'].drop_duplicates(),
                                 new_nodes, new_labels,
                                 1 if y_abnormal > 0 else 0)
            continue

        # 替换逻辑判定
        para_n = logs[(logs['EventTemplate'] == node) & (logs['Label'] == '-')]['ParameterList'].drop_duplicates()
        para_a = logs[(logs['EventTemplate'] == node) & (logs['Label'] != '-')]['ParameterList'].drop_duplicates()
        flag, k = replace_by_param(logs, node, para_n, para_a, new_nodes, new_labels, 1 if y_abnormal > 0 else 0)

        if not flag:
            level_n = logs[(logs['EventTemplate'] == node) & (logs['Label'] == '-')]['Level'].drop_duplicates()
            level_a = logs[(logs['EventTemplate'] == node) & (logs['Label'] != '-')]['Level'].drop_duplicates()
            if should_split_by_level(level_n, level_a):
                split_by_level(logs, node, 'Label', new_nodes, new_labels, nodes)
                replace_map[node] = -1
            else:
                new_nodes.append(node)
                new_labels.append(1 if y_abnormal > 0 else 0)
        else:
            replace_map[node] = k

    return np.array(new_nodes), np.array(new_labels, dtype=int), replace_map