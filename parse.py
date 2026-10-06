#!/bin/python

import sys
import argparse

from collections import defaultdict
from enum import Enum
import numpy as np

from time import sleep

'''
 Preprocessing
1. Convert to individual lines from source format (ie. zipped)
2. Remove malformed lines (ie. non-printable characters)
3. Split messages on whitespace

 Message Type Generation Pipeline
1. Group messages into paritions based on length
2. Run LCS between the first two lines
3. The LCS is that partitions message type
'''

def LCS(inx: list, iny: list):
    # returns a mask of matching values

    CellType = Enum('CellType', [
        ('none', 0),
        ('match', 1),
        ('up', 2),
        ('left', 3),
    ])

    # makes a grid to store info
    #   i n x
    # i _ _ _
    # n _ _ _
    # y _ _ _

    # defaultdict with tuple vector indices and a default value to make later code cleaner
    grid: dict[tuple[int, int], tuple[int, CellType]] = defaultdict(lambda: (0, CellType.left))

    corner = [0, 0]
    for y, y_val in enumerate(iny):
        for x, x_val in enumerate(inx):
            if (x_val == y_val and
                corner[0] <= x and
                corner[1] <= y
            ):
                grid[x, y] = grid[x-1, y-1][0] + 1, CellType.match
                corner = [x + 1, y + 1]
            else:
                if grid[x-1, y][0] > grid[x, y-1][0]:
                    grid[x, y] = grid[x-1, y][0], CellType.left
                else:
                    grid[x, y] = grid[x, y-1][0], CellType.up

    # print the grid (types)
    #CellTypeIcons = [' ', '\\', '^', '<']
    #print('  ' + ''.join(inx))
    #for y, y_val in enumerate(iny):
    #    print(y_val + ' ' + ''.join([CellTypeIcons[grid[x, y][1].value] for x in range(len(inx))]))

    # backtrack from bottom right
    x_matches = []
    y_matches = []
    pos = (len(inx)-1, len(iny)-1)
    while pos[0] >= 0 and pos[1] >= 0:
        print(pos)
        sleep(0.5)
        match grid[pos][1].value:
            case CellType.left.value:
                pos = (pos[0]-1, pos[1])
                x_matches.append(False)
            case CellType.up.value:
                pos = (pos[0], pos[1]-1)
                y_matches.append(False)
            case CellType.match.value:
                pos = (pos[0]-1, pos[1]-1)
                x_matches.append(True)
                y_matches.append(True)
    x_matches.reverse()
    y_matches.reverse()

    return (x_matches, y_matches)

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
