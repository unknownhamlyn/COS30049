#!/bin/python

import sys
import argparse

from collections import defaultdict
from itertools import permutations

import numpy as np

def LCS(inx, iny, debug=False):
    # returns a mask of matching values

    # makes a grid to store info
    #   i n x
    # i _ _ _
    # n _ _ _
    # y _ _ _

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
    x_matches = []
    y_matches = []
    corner = [len(inx)-1, len(iny)-1]
    while -1 not in corner:
        match grid[tuple(corner)]:
            case (-1, 0):
                corner[0] -= 1
                x_matches.insert(0, False)
            case (0, -1):
                corner[1] -= 1
                y_matches.insert(0, False)
            case (-1, -1):
                corner = [i - 1 for i in corner]
                x_matches.insert(0, True)
                y_matches.insert(0, True)

    x_matches = [False]*(1+corner[0]) + x_matches
    y_matches = [False]*(1+corner[1]) + y_matches

    return (x_matches, y_matches)

# O(n^2)
def Template(lines):
    assert len(set([len(line) for line in lines])) == 1, 'lines of differing length'

    mask = np.full(len(lines[0]), True)
    for a, b in permutations(lines, 2):
        mask = np.logical_and(mask, LCS(a, b)[0])
    
    return [mask[k] and v or '*' for k, v in enumerate(lines[0])]

class Parser:
    bins: dict[int, list[list[str]]] = {}
    
    def parse(self, msg: list[str]):
        print(msg)
        if len(msg) not in self.bins:
            self.bins[len(msg)] = [msg]
        else:
            print("".join(set(self.bins[len(msg)][0]) & set(msg)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("file", type=argparse.FileType("r")) # ty: ignore[deprecated]
    args = argparser.parse_args()

    parser = Parser()
    for line in args.file:
        parser.parse(line.split())

    print()
    print(parser.bins)

lines = list(map(lambda x: x.split(), [
    "cpu_freq 34231 3431 cool thumbsup",
    "cpu_freq 23141 592 cool thumbdown",
    "cpu_freq w952 23553 cool haiii"
]))
