#!/usr/bin/env python
from CONSTANTS import *
import sys
import os
# sys.path.append('../../')
from Drain import LogParser
format = {
    'BGL' : '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    'BGL_2k' : '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    'HDFS' : '<Label> <Pid> <Id> <Level> <Component> <Content>',
    'Liberty' : '<Label> <Timestamp> <Date> <Node> <Month> <Day> <Time> <Component> <Content>',
    'Spirit' : '<Label> <Timestamp> <Date> <Node> <Month> <Day> <Time> <Component> <Content>',
    'Thunderbird' : '<Label> <Pid> <Date> <Node> <Month> <Day> <Time> <Component> <Node> <Content>',
    'hades': '<Label> <Content>'
}

def parse(file, dataset, mode, path):
    log_format = format[dataset] # HDFS log format
    # Regular expression list for optional preprocessing (default: [])
    regex = [
        r'blk_(|-)[0-9]+',  # block id
        r'(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])\\.(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])\\.(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])\\.(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5]):(6[0-5]{2}[0-3][0-5]|[1-5]\\d{4}|[1-9]\\d{1,3}|[0-9])',
        # IP
        r'((?<=[^A-Za-z0-9])|^)(\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})((?=[^A-Za-z0-9])|$)',  # Numbers
        r'((?<=[^A-Za-z0-9])|^)([0-9a-f]{6,} ?){3,}((?=[^A-Za-z0-9])|$)',
        r'((?<=[^A-Za-z0-9])|^)([0-9A-F]{4} ?){4,}((?=[^A-Za-z0-9])|$)',
        r'((?<=[^A-Za-z0-9])|^)(0x[a-f0-9A-F]+)((?=[^A-Za-z0-9])|$)',
        r'((?<=[^A-Za-z0-9])|^)([\\-\\+]?\\d+)((?=[^A-Za-z0-9])|$)',
        r'[^:]+:[0-9a-fA-F]{8}',
        r'([0-9]+)',
        # r'<.*>',
        # r'((25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d)))\.){3}(25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d)))',
        # r'\/([\w\.]+\/?)*'
        # r'(?<=executed cmd )(\".+?\")',
        # r'(:[A-Za-z0-9]{8})',
        # r'(00[A-Za-z0-9]{6})',
        # r'core\.\d+',
    ]
    indir = path
    fileName = dataset + ('_' if mode != '' else '') + mode + '.log'
    outdir = os.path.join(path, mode)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    st         = 0.5  # Similarity threshold
    depth      = 4  # Depth of all leaf nodes
    parser = LogParser(log_format, indir=indir, outdir=outdir, depth=depth, st=st, rex=regex)
    parser.parse(fileName)
