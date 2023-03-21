import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import networkx as nx
import os
import sys
import json
import random
import re
from math import ceil, floor, log, log2, log10, sqrt, exp, factorial, gcd, lcm, pi, e, inf, nan, isinf, isnan, isfinite, copysign, fmod, frexp, ldexp, modf, trunc, erf, erfc, gamma, lgamma, hypot, atan2, pow, atan, asin, acos, tan, sin, cos, sinh, cosh, tanh, asinh, acosh, atanh, degrees, radians, expm1, log1p, exp2, log2, fsum, gcd, lcm, tau, inf, nan, isinf, isnan, isfinite, copysign, fmod, frexp, ldexp, modf, trunc, erf, erfc, gamma, lgamma, hypot, atan2, pow, atan, asin, acos, tan, sin, cos, sinh, cosh, tanh, asinh, acosh, atanh, degrees, radians, expm1, log1p, exp2, log2, fsum, gcd, lcm, tau, inf, nan, isinf, isnan, isfinite, copysign, fmod, frexp, ldexp, modf, trunc, erf, erfc, gamma, lgamma, hypot, atan2, pow, atan, asin, acos, tan, sin, cos, sinh, cosh, tanh, asinh, acosh, atanh, degrees, radians, expm1, log1p, exp2, log2, fsum, gcd, lcm, tau, inf, nan, isinf, isnan, isfinite, copysign, fmod, frexp, ldexp, modf, trunc, erf, erfc, gamma, lgamma, hypot, atan2, pow, atan, asin, acos, tan, sin, cos, sinh, cosh, tanh, asinh, acosh, atanh, degrees, radians, expm1, log1p, exp2, log2, fsum, gcd, lcm, tau, inf, nan, isinf, isnan, isfinite, copysign, fmod, frexp, ldexp, modf, trunc, erf, erfc, gamma, lgamma, hypot, atan2
from collections import Counter, defaultdict, OrderedDict, namedtuple, deque
from functools import partial, partialmethod, reduce, wraps, cache, lru_cache, cached_property, singledispatch, singledispatchmethod
from itertools import count, cycle, product as cartesian_product, permutations, combinations, combinations_with_replacement, accumulate, starmap
from tqdm import tqdm # from tqdm.notebook import tqdm
from uuid import uuid4
from datetime import datetime, timedelta
from time import time, sleep
from toolz import memoize, curry, diff, unique, valmap, valfilter, itemmap, itemfilter, keymap, keyfilter, merge_sorted, interleave, isdistinct, diff, peek, peekn, countby, juxt, excepts, merge, merge_with, assoc, dissoc
from more_itertools import unzip, chunked, chunked_even, minmax, filter_except, numeric_range, make_decorator,replace, locate,countable,unique_everseen, always_iterable,unique_justseen,map_except,count_cycle, mark_ends, sample, distribute, bucket, peekable, seekable,spy,transpose, sieve,polynomial_from_roots,flatten, intersperse, partition, powerset, collapse, split_at, flatten,split_before, split_after, split_when, take



def tuple_eq(tup1,tup2):
    # checks if two tuples have equal values
    return len(set(tup1).union(set(tup2))) == len(tup1) == len(tup2)
    
def tuple_rev(tup):
    # reverses a tuple
    return (tup[1],tup[0])

def tuple_int(tup1,tup2):
    # checks if two tuples intersect
    return len(set(tup1).union(set(tup2))) < len(tup1) + len(tup2)

def tuple_sum(tup1,tup2):
    # Adds values of two tuples
    return tuple(map(sum,zip(tup1,tup2)))


identity_fn = lambda x: x
constant_fn = lambda x: lambda y: x
neg = lambda f: lambda x: -f(x)
sign = lambda x: 1 if x > 0 else -1 if x < 0 else 0

def euclidean_dist(node1,node2):
    return np.sqrt((node1.x-node2.x)**2 + (node1.y-node2.y)**2)

def manhattan_dist(node1,node2):
    return abs(node1.x-node2.x) + abs(node1.y-node2.y)



class Node:
    def __init__(self, loc):
        self.x = loc[0]
        self.y = loc[1]
        self.loc = loc
        self.id = uuid4().hex
        self._weight = constant_fn(None)
        
    @property
    def weight(self):
        return self._weight(self)
    @weight.setter
    def weight(self, wf):
        if callable(wf):
            self._weight = wf
        elif isinstance(wf, (int,float)):
            self._weight = constant_fn(wf)
        else:
            raise TypeError('weight must be a function or a number')
    
    def __repr__(self) -> str:
        if self.x == None or self.y == None:
            return f'Node: ({self.x},{self.y})'
        string = f'Node: ({self.x:.2f},{self.y:.2f})'
        if not callable(self.weight) and self.weight != None:
            string += f'||weight {self.weight:.2f}'
        return string
    def __str__(self) -> str:
        return self.__repr__()
    def __eq__(self, o: object) -> bool:
        return self.loc == o.loc
    def __hash__(self) -> int:
        return hash(self.loc)
    def __getitem__(self, key):
        return self.loc[key]
    def __iter__(self):
        return iter(self.loc)
    def __contains__(self, item):
        return item in self.loc
    def __abs__(self):
        return np.sqrt(self.x**2 + self.y**2)
    
    @classmethod
    def make(cls, x, y, weight=constant_fn(None)):
        node = cls((x,y))
        node.weight = weight
        return node
    @staticmethod
    def dist(node1,node2,metric='manhattan'):
        if metric == 'euclidean':
            return euclidean_dist(node1,node2)
        elif metric == 'manhattan':
            return manhattan_dist(node1,node2)
        else:
            return metric(node1,node2)
        
    def dist_to(self, node2, metric='manhattan'):
        return Node.dist(self,node2,metric)
    
    
    def move(self, x, y, relative=True):
        if not relative:
            self.x = x
            self.y = y
        self.x += x
        self.y += y

class Edge:
    def __init__(self, node1, node2, weight= lambda x : manhattan_dist(x.node1,x.node2)):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight
        self.id = uuid4().hex
        
    @property
    def weight(self):
        return self._weight(self)
    @weight.setter
    def weight(self, wf):
        if callable(wf):
            self._weight = wf
        elif isinstance(wf, (int,float)):
            self._weight = constant_fn(wf)
        else:
            raise TypeError('weight must be a function or a number')


    @property
    def nodes(self):
        return (self.node1,self.node2)
    @nodes.setter
    def nodes(self, nodes):
        self.node1 = nodes[0]
        self.node2 = nodes[1]
    @property
    def node_ids(self):
        return (self.node1.id,self.node2.id)
    @property
    def node_locs(self):
        return (self.node1.loc,self.node2.loc)
    @property
    def node_weights(self):
        return (self.node1.weight,self.node2.weight)
    
    @classmethod
    def make(cls, node1, node2, weight= lambda x : manhattan_dist(x.node1,x.node2)):
        return cls(node1,node2,weight)

    def __repr__(self) -> str:
        return f'Edge: {self.node1} -> {self.node2} || weight: {self.weight:.2f}'
    def __str__(self) -> str:
        return self.__repr__()
    def __eq__(self, o: object) -> bool:
        return tuple_eq(self.node_locs,o.node_locs)
    def __hash__(self) -> int:
        return hash(self.node_locs)
    def __iter__(self):
        return iter(self.node_locs)
    def __contains__(self, item):
        return item in self.node_locs
    def __getitem__(self, key):
        return self.node_locs[key]
    
    def other_node(self, node):
        if node == self.node1:
            return self.node2
        elif node == self.node2:
            return self.node1
        else:
            raise ValueError('node is not in edge')
        
    def is_loop(self):
        return self.node1 == self.node2
    
class Network:
    def __init__(self, nodes, edges=None):
        self.nodes = nodes
        self.edges = edges if edges else []
        # self.node_ids = {node.id:node for node in self.nodes}
        # self.edge_ids = {edge.id:edge for edge in self.edges}
        # self.node_locs = {node.loc:node for node in self.nodes}
        # self.adjacency = {node.id: dict() for node in self.nodes}
        # for edge in self.edges:
        #     self.adjacency[edge.node1.id][edge.node2.id] = edge.id
        #     self.adjacency[edge.node2.id][edge.node1.id] = edge.id
        
    def translate_node(self, id):
        return self.node_ids[id]
    def translate_nodes(self, ids):
        return list(map(self.translate_node, ids))
    def translate_edge(self, id):
        return self.edge_ids[id]
    def translate_edges(self, ids):
        return list(map(self.translate_edge, ids))
    def get_node(self, loc):
        return self.node_locs[loc]
    def adjacent_nodes(self, node):
        return self.translate_nodes(self.adjacency[node.id].keys())
    def adjacent_edges(self, node):
        return self.translate_edges(self.adjacency[node.id].values())
    def adjacents(self, node):
        return self.adjacent_nodes(node), self.adjacent_edges(node)
    def get_edge(self, node1, node2):
        return self.translate_edge(self.adjacency[node1.id][node2.id])
    
    @property
    def nodes(self):
        return self._nodes
    @nodes.setter
    def nodes(self, nodes):
        self._nodes = list(nodes)
        self.node_ids = {node.id:node for node in self.nodes}
        self.node_locs = {node.loc:node for node in self.nodes}
        if len(self.nodes) != len(self.node_locs):
            raise ValueError('duplicate node ids')
        self.adjacency = {node.id: dict() for node in self.nodes}
        
    @property
    def edges(self):
        return self._edges
    @edges.setter
    def edges(self, edges):
        self._edges = []
        self.edge_ids = {}
        for edge in edges:
            if edge.node1.id not in self.adjacency or edge.node2.id not in self.adjacency:
                raise ValueError('edge nodes not in network')
            if edge.node1.id in self.adjacency[edge.node2.id]:
                continue
            if edge.node2.id in self.adjacency[edge.node1.id]:
                continue
            self._edges.append(edge)
            self.edge_ids[edge.id] = edge
            self.adjacency[edge.node1.id][edge.node2.id] = edge.id
            self.adjacency[edge.node2.id][edge.node1.id] = edge.id


    def __repr__(self,draw = False) -> str:
        if draw:
            self.draw()
        line_break = '=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-='
        return line_break + '\n' + \
                f"Network with {len(self.nodes)} Nodes and {len(self.edges)} Edges\n" + \
                line_break + '\n' + \
                'Nodes:\n' + '\n'.join([node.__repr__() for node in self.nodes]) + '\n' + \
                line_break + '\n' + \
                'Edges:\n' + '\n'.join([str(edge) for edge in self.edges]) + '\n'

    def __str__(self) -> str:
        return self.__repr__()
    def __iter__(self):
        return iter(self.nodes)
    def __contains__(self, item):
        return item in self.nodes
    def __getitem__(self, key):
        return self.nodes[key]
    def __len__(self):
        return len(self.nodes)
    def __eq__(self, o: object) -> bool:
        return self.nodes == o.nodes and self.edges == o.edges
    
    def add_node(self, node):
        if node.id in self.node_ids:
            raise ValueError('node id already in network')
        if node.loc in self.node_locs:
            raise ValueError('node loc already in network')
        self.nodes.append(node)
        self.node_ids[node.id] = node
        self.node_locs[node.loc] = node
        self.adjacency[node.id] = dict()
    def add_edge(self, edge):
        if edge.id in self.edge_ids:
            raise ValueError('edge id already in network')
        if edge.node1.id not in self.node_ids:
            raise ValueError('edge node1 not in network')
        if edge.node2.id not in self.node_ids:
            raise ValueError('edge node2 not in network')
        if edge.node1.id in self.adjacency[edge.node2.id]:
            raise ValueError('edge nodes already connected')
        if edge.node2.id in self.adjacency[edge.node1.id]:
            raise ValueError('edge nodes already connected')
        self.edges.append(edge)
        self.edge_ids[edge.id] = edge
        self.adjacency[edge.node1.id][edge.node2.id] = edge.id
        self.adjacency[edge.node2.id][edge.node1.id] = edge.id
        
    def remove_node(self, node):
        if node.id not in self.node_ids:
            raise ValueError('node id not in network')
        if node.loc not in self.node_locs:
            raise ValueError('node loc not in network')
        for edge in self.adjacent_edges(node):
            self.remove_edge(edge)
        self.nodes.remove(node)
        del self.node_ids[node.id]
        del self.node_locs[node.loc]
        del self.adjacency[node.id]
        
    def remove_edge(self, edge):
        if edge.id not in self.edge_ids:
            raise ValueError('edge id not in network')
        if edge.node1.id not in self.node_ids:
            raise ValueError('edge node1 not in network')
        if edge.node2.id not in self.node_ids:
            raise ValueError('edge node2 not in network')
        if edge.node1.id not in self.adjacency[edge.node2.id]:
            raise ValueError('edge nodes not connected')
        if edge.node2.id not in self.adjacency[edge.node1.id]:
            raise ValueError('edge nodes not connected')
        self.edges.remove(edge)
        del self.edge_ids[edge.id]
        del self.adjacency[edge.node1.id][edge.node2.id]
        del self.adjacency[edge.node2.id][edge.node1.id]
        
    
    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)
    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge)
    def remove_nodes(self, nodes):
        for node in nodes:
            self.remove_node(node)
    def remove_edges(self, edges):
        for edge in edges:
            self.remove_edge(edge)
    
    def move_node(self, node, loc, relative=True):
        if node.id not in self.node_ids:
            raise ValueError('node id not in network')
        if node.loc not in self.node_locs:
            raise ValueError('node loc not in network')
        if relative:
            loc = tuple_sum(node.loc, loc)
        if loc in self.node_locs:
            raise ValueError('node loc already in network')
        del self.node_locs[node.loc]
        self.node_locs[loc] = node
        node.loc = loc

    def reach(self, node, max_dist = None):
        if node.id not in self.node_ids:
            raise ValueError('node id not in network')
        if node.loc not in self.node_locs:
            raise ValueError('node loc not in network')
        if max_dist is None:
            max_dist = float('inf')
        visited = set()
        queue = deque([(node, 0)])
        while queue:
            node, dist = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if dist >= max_dist:
                continue
            for edge in self.adjacent_edges(node):
                queue.append((edge.other_node(node), dist + edge.weight))
        return visited

    def subtnetwork(self, node):
        reach = self.reach(node)
        edges = set()
        for node in reach:
            edges.update(self.adjacent_edges(node))
        return Network(reach, edges)
    
    def subnetworks(self):
        visited = set()
        subnetworks = []
        for node in self.nodes:
            if node in visited:
                continue
            subnetworks.append(self.subtnetwork(node))
            visited.update(subnetworks[-1])
        return subnetworks
    
    def find_path(self,node1, node2, func = lambda x : x[2]):
        """finds a path between two nodes using a function to sort the queue
        
        func should take a tuple of (node, path, dist) and return a value to sort by
        default sorts by path-weight
        """
        if node1 not in self.nodes or node2 not in self.nodes:
            raise ValueError('node not in network')
        if node1 == node2:
            return [node1]
        visited = set()
        queue = deque([(node1, [node1], 0)]) #node, path, dist
        while queue:
            next, path, dist = queue.popleft()
            if next in visited:
                continue
            visited.add(next)
            for edge in self.adjacent_edges(next):
                new_path = path + [edge.other_node(next)]
                if edge.other_node(next) == node2:
                    return new_path
                queue.append((edge.other_node(next), new_path, dist + edge.weight))
                queue = deque(sorted(queue, key=func))
        return None
    
    
    
                # for edge in self.adjacent_edges(next):
                # new_path = path + [edge.other_node(next)]
                # if edge.other_node(next) == node2:
                #     return new_path
                # queue.append((edge.other_node(next), new_path))
                # if func:
                #     queue = deque(sorted(queue, key=lambda x: func(x[0])))
    def fill_path_edges(self, path):
        if len(path) < 2:
            return []
        edges = []
        for i in range(len(path[1:])):
            edge = self.get_edge(path[i], path[i+1])
            edges.append(edge)
        return edges
    
    
    def path_weight(self, edges):
        return sum([edge.weight for edge in edges])
    
    def path_weight_optimal(self, node1, node2):
        return self.path_weight(self.fill_path_edges(self.find_path(node1, node2)))
    
    def full_path(self, node1, node2, func = lambda x : x[2]):
        nodes = self.find_path(node1, node2, func)
        if nodes is None:
            return None
        edges = self.fill_path_edges(nodes)
        weight = self.path_weight(edges)
        return nodes, edges, weight, len(nodes)
    
