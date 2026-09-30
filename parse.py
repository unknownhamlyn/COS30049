#!/bin/python

import sys
import argparse

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

# O(n*m) but fuck you
def LCS(in1: list[str], in2: list[str]):
    out: list[str] = []
    for i in in1:
        for j in in2:
            if i == j:
                out.append(i)
                break
    return len(out)

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
