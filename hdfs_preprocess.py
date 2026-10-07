import argparse
import pandas as pd
import numpy as np
import sklearn as sk

from parse import ExtractTemplateBins

def load_lines(file):
    lines = file.readlines()
    # remove lines that contain non printable characters
    lines = np.array(filter(lambda x: x.printable, lines))
    
    sorted_lines = ExtractTemplateBins(lines)

    raise Exception('not implimented')
    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("file", type=argparse.FileType("r")) # ty: ignore[deprecated]
    args = argparser.parse_args()

    df = load_lines(args.file)
    
