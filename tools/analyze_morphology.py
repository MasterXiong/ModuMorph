import networkx as nx
import argparse
import os
import sys
import pickle
import numpy as np
from multiprocessing import Pool

from metamorph.algos.ppo.ppo import PPO
from metamorph.config import cfg
from metamorph.config import dump_cfg
from metamorph.utils import file as fu
from metamorph.utils import sample as su
from metamorph.utils import sweep as swu
from metamorph.algos.ppo.envs import make_env

train_xml_folder = 'unimals_100/train/xml'
test_xml_folder = 'unimals_100/test/xml'

    
with open('unimals_100/train/graph.pkl', 'rb') as f:
    train_graphs = pickle.load(f)
with open('unimals_100/test/graph.pkl', 'rb') as f:
    test_graphs = pickle.load(f)


def compute_graph_dist(i):
    
    score = np.zeros(100)
    test_graph = test_graphs[i]
    for j in range(100):
        train_graph = train_graphs[j]
        x = nx.optimize_edit_paths(test_graph, train_graph, roots=(0, 0))
        for vertex_path, edge_path, cost in x:
            score[j] = cost
        print (i, j, score[j])

    with open('unimals_100/graph_edit_dist_%d.pkl' %(i), 'wb') as f:
        pickle.dump(score, f)

'''
with Pool(10) as p:
    p.map(compute_graph_dist, list(range(100)))
'''


# merge results
print (os.listdir(test_xml_folder))
print (len(os.listdir(test_xml_folder)))
'''
results = np.zeros([100, 100])
for i in range(100):
    with open('unimals_100/graph_edit_dist_%d.pkl' %(i), 'rb') as f:
        results[i] = pickle.load(f)
with open('unimals_100/graph_edit_dist.pkl', 'wb') as f:
    pickle.dump(results, f)
'''

