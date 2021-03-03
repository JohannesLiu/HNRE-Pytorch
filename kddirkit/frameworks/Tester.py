import torch
import numpy as np
import pandas as pd

class Tester(object):
    def __init__(self):
        super(Tester, self).__init__()
        f = open("raw_data/relation2id.txt", "r")
        content = f.readlines()[1:]
        self.id2rel = {}
        for i in content:
            rel, rid = i.strip().split()
            self.id2rel[(int)(rid)] = rel
        f.close()

        self.fewrel_100 = {}
        f = open("data/rel100.txt", "r")
        content = f.readlines()
        for i in content:
            self.fewrel_100[i.strip()] = 1
        f.close()

        self.fewrel_200 = {}
        f = open("data/rel200.txt", "r")
        content = f.readlines()
        for i in content:
            self.fewrel_200[i.strip()] = 1
        f.close()