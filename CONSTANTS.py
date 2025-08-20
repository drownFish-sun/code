import os

import numpy as np

PROJECT_ROOT = os.path.abspath(os.getcwd())

# times = []
# acc, pre, re, f1 = [], [], [], []
# with open('res.txt', 'r') as f:
#     lines = f.readlines()
#     for i, line in enumerate(lines):
#         if i & 1 == 0:
#             times.append(float(line.strip()))
#         else:
#             v = line.strip().split(' ')
#             acc.append(float(v[1].strip(',')))
#             pre.append(float(v[3].strip(',')))
#             f1.append(float(v[5].strip(',')))
#         if len(times) == 10:
#             print(np.mean(times), np.std(times))
#             print(np.mean(acc), np.std(acc))
#             print(np.mean(pre), np.std(pre))
#             print(np.mean(f1), np.std(f1))
#             times, acc, pre, f1 = [], [], [], []
