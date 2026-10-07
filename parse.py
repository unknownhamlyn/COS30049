#!/bin/python

import sys
import argparse

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn

def LCS(inx, iny, debug=False):
    # returns a mask of matching values

    grid = defaultdict(lambda: (-1, 0))

    corner = [-1, -1]
    for y, y_val in enumerate(iny):
        for x, x_val in enumerate(inx):
            if (x_val == y_val and
                corner[0] < x and
                corner[1] < y
            ):
                grid[x, y] = (-1, -1)
                corner = [x, y]
            else:
                if grid[x, y-1][1] == -1:
                    grid[x, y] = (0, -1)

    # print the grid (types)
    if debug:
        icons = {
            (-1, 0): '<',
            (0, -1): '^',
            (-1, -1): '\\'
        }
        xLabelWidth = max([len(str(i)) for i in inx])+1
        yLabelWidth = max([len(str(i)) for i in iny])
        print(' '*yLabelWidth + ''.join([str(i).rjust(xLabelWidth, ' ') for i in inx]))
        for y, y_val in enumerate(iny):
            print(str(y_val).rjust(yLabelWidth, ' ') + ''.join([icons[grid[x, y]].rjust(xLabelWidth, ' ') for x in range(len(inx))]))

    # backtrack from bottom right
    x_matches, y_matches = np.full(len(inx), False), np.full(len(iny), False)
    pos = [len(inx)-1, len(iny)-1]
    while -1 not in pos:
        move = grid[tuple(pos)]
        if move ==(-1, -1):
            x_matches[pos[0]] = True
            y_matches[pos[1]] = True
        pos = np.add(pos, move)

    return (x_matches, y_matches)

def MatchesTemplate(line, template):
    if type(template == str):
        template = template.split()
    return all([l == t or t == '<*>' for l, t in zip(line, template)])

def Distance(a, b):
    return list(LCS(a, b)[0]).count(True) / max(len(a), len(b))

def ExtractTemplatesDbscan(lines):
    if type(lines) == list:
        lines = np.array(list, dtype=object)

    # compute metric matrix
    count = len(lines)
    metric = np.zeros((count, count))

    # consider sparse metric matrix?
    # 200  -  3.7s
    # 500  -  7.9s
    # 1000 - 24.5s
    for pos in [(x, y) for y in range(1, count) for x in range(0, y)]:
        metric[pos] = Distance(*lines[list(pos)])

    db = sklearn.cluster.DBSCAN(
        metric="precomputed",
        eps=0.15,
        min_samples=10
    ).fit_predict(metric)

    raise Exception('not implimented')
    #return pd.DataFrame({'lines': lines, 'type': db})

def LinesToTemplate(a, b):
    return ' '.join([unique and field or '<*>' for field, unique in zip(a, LCS(a, b)[0])])

def ExtractTemplateBins(lines):
    print(len(lines))
    # bin(template, length, records)
    @dataclass
    class Bin:
        length: int
        template: str | None
        records: list[list[str]]

    bins: list[Bin] = []

    for line in lines:
        length = len(line)
        sorted_line = False
        for bin in bins:
            if bin.template and MatchesTemplate(line, bin.template):
                bin.records.append(line)
                sorted_line = True
                break
            elif bin.template is None and bin.length == length and bin.records:
                bin.template = LinesToTemplate(line, bin.records[0])
                bin.records.append(line)
                sorted_line = True
                break
        if not sorted_line:
            bins.append(Bin(length, None, [line])) 
    return bins

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("file", type=argparse.FileType("r")) # ty: ignore[deprecated]
    args = argparser.parse_args()

with open('data/HDFS.log') as f:
    lines = np.array([next(f).split() for _ in range(20000)], dtype=object)
