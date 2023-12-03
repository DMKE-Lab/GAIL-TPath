from __future__ import division
import tensorflow as tf
import numpy as np
import collections
from itertools import count
import sys
from BFS.KB import KB
from BFS.BFS import BFS
import time

dataPath = ''
graphPath = dataPath + 'train.txt'
relationPath = dataPath + 'train.txt'

def sampling(path_threshold=2, path=None):
    f = open(path)
    content = f.readlines()
    f.close()
    kb = KB()
    for line in content:
        elements = line.strip().split("\t")
        kb.addRelation(elements[0], elements[1], elements[2])
        kb.addRelation(elements[2], elements[1], elements[0])
    f = open(relationPath)
    train_data = f.readlines()
    f.close()
    num_samples = len(train_data)
    demo_path_dict = {}
    for episode in range(num_samples):
        sample = train_data[episode % num_samples].split()
        ent1 = sample[0]
        rel = sample[1]
        ent2 = sample[2]
        kb.removePath(ent1, ent2)
        try:
            suc, entity_list, path_list = BFS(kb, ent1, ent2)
            path_str = ' -> '.join(path_list)
        except Exception as e:
            print('Episode %d' % episode)
            print('Cannot find a path')
            continue
        if path_str not in demo_path_dict:
            demo_path_dict[path_str] = 1
        else:
            demo_path_dict[path_str] += 1
        if rel not in demo_path_dict:
            demo_path_dict[rel] = 1
        else:
            demo_path_dict[rel] += 1
        kb.addRelation(ent1, rel, ent2)
    demo_path_dict = {k: v
                      for k, v in demo_path_dict.items()
                      if v >= path_threshold}
    demo_path_list = sorted(demo_path_dict.items(), key=lambda x: x[1], reverse=True)
    print('BFS found paths:', len(demo_path_list))
    f = open(dataPath + 'demo_path.txt', 'w')
    for item in demo_path_list[:5]:
        f.write(item[0] + '\n')
    f.close()
    print('demo path saved')
    f = open(dataPath + 'demo_path_stat.txt', 'w')
    for item in demo_path_list:
        f.write(item[0] + '\t' + str(item[1]) + '\n')
    f.close()
    print('demo path stat saved')
    return

if __name__ == '__main__':
    sampling(path_threshold=2, path=graphPath)
