import numpy as np
import networkx as nx
import pylab as plt
from scipy.stats import norm,uniform,binom,poisson
import random
from functools import reduce
from itertools import product
from collections import deque





def float_range(stop, start=0, step=1):
    while start < stop:
        yield float(start)
        start += step 

def tuple_eq(tup1,tup2):
    return len(set(tup1).union(set(tup2))) == len(tup1) == len(tup2)
    
def tuple_rev(tup):
    return (tup[1],tup[0])

def tuple_int(tup1,tup2):
    return len(set(tup1).union(set(tup2))) < len(tup1) + len(tup2)

class Node:
    def __init__(self, coord:tuple[float], weight:float = 0) -> None:
        self.coord = coord # tuple of coordinates
        self.weight = weight # weight of node
        
    def __repr__(self) -> str:
        smalls = tuple(f"{x:.2f}" for x in self.coord)
        return f"Node{smalls}" + ", Weight{self.weight:.2f})"*bool(self.weight)
    
    def __str__(self) -> str:
        smalls = tuple(f"{x:.2f}" for x in self.coord)
        return f"{smalls}" + ", {self.weight:.2f}"*bool(self.weight)
    
    def __eq__(self, other):
        return self.coord == other.coord
    
    def __hash__(self):
        return hash(self.coord)
    
    def __lt__(self, other):
        return self.weight < other.weight
    
    def __gt__(self, other):
        return self.weight > other.weight
    
    def __le__(self, other):
        return self.weight <= other.weight
    
    def __ge__(self, other):
        return self.weight >= other.weight

    def __iter__(self):
        return iter(self.coord)
    
    def __getitem__(self, key):
        return self.coord[key]
    
    def __abs__(self):
        return np.sqrt(reduce(lambda a,b : a+b ,map(lambda x: x ** 2, self.coord)))
    
    def __len__(self):
        return len(self.coord)
    
    def __contains__(self, item):
        return item in self.coord
    
    #############################################
    #############################################    
    #############################################
    # Properties
    #############################################
    #############################################
    #############################################
    
    @property
    def coord(self):
        return self._coord
    @coord.setter
    def coord(self, value) -> None:
        self._coord = value
        
    @property
    def weight(self):
        return self._weight
    @weight.setter
    def weight(self, value) -> None:
        self._weight = value
    
    #############################################
    #############################################    
    #############################################
    # Methods
    #############################################
    #############################################
    #############################################
    
    @classmethod
    def from_tuple(cls, tup):
        return cls(tup[:-1], tup[-1])
    
    def distance(self, other):
        return np.sqrt(reduce(lambda a,b : a+b ,map(lambda x: (x[0] - x[1]) ** 2, zip(self.coord, other.coord))))
    
    def slope(self, other):
        if self[0] == other[0]:
            return np.inf
        return (other[1] - self[1]) / (other[0] - self[0])
    
    def intercept(self, other):
        if self[0] == other[0]:
            return np.nan
        return self[1] - self.slope(other) * self[0]
    
    def equation(self, other):
        if self[0] == other[0]:
            return f"x = {self[0]}"
        return f"y = {self.slope(other)}x + {self.intercept(other)}"
    
    def area(self, node1, node2):
        return abs((self[0] * (node1[1] - node2[1]) + node1[0] * (node2[1] - self[1]) + node2[0] * (self[1] - node1[1])) / 2)
    
    
class Edge:
    def __init__(self, nodes:tuple, dir:bool = False , weight:float = 1, key:int = 0):
        self.nodes = nodes # tuple of two nodes
        self.weight = weight # weight of the edge
        self.dir = dir # directed or undirected
        self.key = key # for multiedges
        
    def __repr__(self) -> str:
        if self.dir:
            return f"Edge({self.nodes[0]} -> {self.nodes[1]}" + f", Weight {self.weight})" + f" with Key {self.key}"*(self.key != 0)
        return f"Edge({self.nodes[0]} <-> {self.nodes[1]}" + f", Weight {self.weight})" + f" with Key {self.key}"*(self.key != 0)
    
    def __str__(self) -> str:
        if self.dir:
            return f"{self.nodes[0]} -> {self.nodes[1]}" + f" || {self.weight}" + f" || {self.key}"*(self.key != 0)
        return f"{self.nodes[0]} <-> {self.nodes[1]}" + f" || {self.weight}" + f" || {self.key}"*(self.key != 0)

    def __eq__(self, other):
        if self.dir:
            return self.nodes == other.nodes and self.dir == other.dir and self.key == other.key
        else:
            return set(self.nodes) == set(other.nodes) and self.dir == other.dir and self.key == other.key
    
    def __hash__(self):
        return hash((self.nodes, self.dir, self.key))
    
    def __lt__(self, other):
        if self.nodes != other.nodes:
            raise ValueError("Edges are not comparable")
        return self.weight < other.weight and self.nodes == other.nodes and self.dir == other.dir
    
    def __gt__(self, other):
        if self.nodes != other.nodes:
            raise ValueError("Edges are not comparable")
        return self.weight > other.weight and self.nodes == other.nodes and self.dir == other.dir
    
    def __le__(self, other):
        if self.nodes != other.nodes:
            raise ValueError("Edges are not comparable")
        return self.weight <= other.weight and self.nodes == other.nodes and self.dir == other.dir
    
    def __ge__(self, other):
        if self.nodes != other.nodes:
            raise ValueError("Edges are not comparable")
        return self.weight >= other.weight and self.nodes == other.nodes and self.dir == other.dir
    
    def __iter__(self):
        return iter(self.nodes)
    
    def __getitem__(self, key):
        return self.nodes[key]
    
    def __abs__(self):
        return self.nodes[0].distance(self.nodes[1])
    
    def __neg__(self):
        if self.dir:
            return Edge(tuple_rev(self.nodes), self.dir, self.weight, self.key)
        else:
            raise ValueError("Undirected Edge")
        
    def __contains__(self, item) -> bool:
        return item in self.nodes
    
    def __len__(self):
        return 1
    
    #############################################
    #############################################    
    #############################################
    # Properties
    #############################################
    #############################################
    #############################################
    
    @property
    def nodes(self):
        return self._nodes
    @nodes.setter
    def nodes(self, value):
        if len(value) != 2:
            raise ValueError("Edge must have two nodes")
        self._nodes = value
    
    @property
    def weight(self):
        return self._weight
    @weight.setter
    def weight(self, value):
        self._weight = value
        
    @property
    def dir(self):
        return self._dir
    @dir.setter
    def dir(self, value):
        self._dir = value
        
    @property
    def key(self):
        return self._key
    @key.setter
    def key(self, value):
        self._key = value
        
    #############################################
    #############################################    
    #############################################
    # Methods
    #############################################
    #############################################
    #############################################
    
    @classmethod
    def new_edge(cls, node1, node2, dir:bool = False, weight:float = 1, key:int = 0):
        return cls((node1, node2), dir, weight, key)
    
    @classmethod
    def reverse_edge(cls, edge):
        if not edge.dir:
            raise ValueError("Undirected Edge")
        elif cls(tuple_rev(edge.nodes), edge.dir, edge.weight, edge.key) in edge.nodes[1].edges:
            return cls.reverse_edge(cls(edge.nodes, edge.dir, edge.weight, edge.key+1))
        return cls(tuple_rev(edge.nodes), edge.dir, edge.weight, edge.key)
    
    def other(self, node):
        if node not in self.nodes:
            raise ValueError("Node not in Edge")
        return self.nodes[1] if self.nodes[0] == node else self.nodes[0]
    
    def extremes(self):
        return self.nodes
    
    def is_loop(self):
        return self.nodes[0] == self.nodes[1]
    
    def is_multi(self, other):
        return tuple_eq(self.nodes,other.nodes) and self.dir == other.dir
    
    def is_parallel(self, other):
        if self.dir and other.dir:
            return self.nodes == other.nodes
        elif self.dir or other.dir:
            return False
        else:
            return tuple_eq(self.nodes,other.nodes)
        
    def is_anti_parallel(self, other):
        if self.dir and other.dir:
            return self.nodes == tuple_rev(other.nodes)
        elif self.dir or other.dir:
            return False
        else:
            return tuple_eq(self.nodes,other.nodes)
        
    def is_adjacent(self, other) -> bool:
        return tuple_int(self.nodes,other.nodes)

    def flip(self):
        return -self
    
    
class Network:
    def __init__(self, nodes = None, edges = None):
        self.nodes = nodes if nodes else []
        self.edges = edges if edges else []
        self.adjacency = { node : [] for node in self.nodes }
        self.edge_adjacency = {node : [] for node in self.nodes}
        for edge in self.edges:
            self.adjacency[edge[0]].append(edge[1])
            self.edge_adjacency[edge[0]].append(edge)
            if not edge.dir:
                self.adjacency[edge[1]].append(edge[0])
                self.edge_adjacency[edge[1]].append(edge)
        
    
    def __repr__(self,draw = False) -> str:
        if draw:
            self.draw()
        line_break = '=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-='
        return line_break + '\n' + \
                f"Network with {len(self.nodes)} Nodes and {len(self.edges)} Edges\n" + \
                line_break + '\n' + \
                'Nodes:\n' + '\n'.join([str(node) for node in self.nodes]) + '\n' + \
                line_break + '\n' + \
                'Edges:\n' + '\n'.join([str(edge) for edge in self.edges]) + '\n'
                
    def __str__(self) -> str:
        return self.__repr__(draw=False)
    
    def __iter__(self):
        return iter(self.nodes)
    
    def __getitem__(self, key ,  component = 'node'):
        if component == 'node':
            return self.nodes[key]
        elif component == 'edge':
            return self.edges[key]
        else:
            raise ValueError("Component must be 'node' or 'edge'")
    
    def __len__(self):
        return len(self.nodes)
    
    def __contains__(self, item) -> bool:
        try:
            return item in self.nodes
        except:
            try:
                return item in self.edges
            except:
                False

    def __hash__(self) -> int:
        return hash((self.nodes, self.edges))
    
    def __eq__(self, other) -> bool:
        return self.nodes == other.nodes and self.edges == other.edges
    
    def __ne__(self, other) -> bool:
        return self.nodes != other.nodes or self.edges != other.edges
    
    def __add__(self, other):
        return Network(self.nodes + other.nodes, self.edges + other.edges)
    
    def __sub__(self, other):
        return Network([node for node in self.nodes if node not in other.nodes], [edge for edge in self.edges if edge not in other.edges])
    
    #############################################    
    #############################################
    # Properties
    #############################################
    #############################################
    
    @property
    def nodes(self):
        return self._nodes 
    @nodes.setter
    def nodes(self, value):
        self._nodes = value

    @property
    def edges(self):
        return self._edges
    @edges.setter
    def edges(self, value):
        self._edges = value
        
    #############################################    
    #############################################
    # Methods
    #############################################
    #############################################
    # Change/Compare Properties
    #############################################
    
    def get_edges(self, node1, node2):
        return [edge for edge in self.edge_adjacency[node1] if node2 in edge]

    def change_node_weight(self, node, weight):
        if node not in self.nodes:
            raise ValueError("Node not in Network")
        node.weight = weight
        
    def change_node_weights(self, weight_func):
        for node in self.nodes:
            node.weight = weight_func(node)
            
    def change_edge_weight(self, edge, weight):
        if edge not in self.edges:
            raise ValueError("Edge not in Network")
        edge.weight = weight
        
    def change_edge_weights(self, weight_func):
        for edge in self.edges:
            edge.weight = weight_func(edge)
    
    def func_node_compare(self, func, node, reduce = False):
        comparable = self.adjacency[node].copy()
        if reduce:
            return func(comparable)
        return map(func,comparable)
    
    def func_edge_compare(self, func, node1, node2, reduce = False):
        if node1 not in self.nodes or node2 not in self.nodes:
            raise ValueError("Node not in Network")
        elif node2 not in self.adjacency[node1] and node1 not in self.adjacency[node2]:
            raise ValueError("Nodes are not adjacent")
        else:
            edges = self.get_edges(node1,node2)
            if reduce:
                return func(edges)
            else:
                return map(func,edges)
    
    def func_edges(self, func, reduce = False):
        if reduce:
            return func(self.edges)
        else:
            return map(func,self.edges)
    
    #############################################
    # Add/Remove Nodes/Edges
    #############################################
     
    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)
            self.adjacency[node] = []
            self.edge_adjacency[node] = []
        else:
            raise ValueError("Node already in Network")
        
    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)
    
    def add_edge(self, edge):
        if edge not in self.edges:
            self.edges.append(edge)
            self.adjacency[edge[0]].append(edge[1])
            self.edge_adjacency[edge[0]].append(edge)
            if not edge.dir:
                self.adjacency[edge[1]].append(edge[0])
                self.edge_adjacency[edge[1]].append(edge)
        else:
            self.add_edge(Edge(edge.nodes,edge.weight,edge.dir,edge.key + 1))
    
    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge)
            
    def remove_node(self, node):
        if node in self.nodes:
            self.nodes.remove(node)
            opposites = self.adjacency.pop(node)
            for opposite in opposites:
                self.adjacency[opposite].remove(node)
                self.edge_adjacency[opposite] = [edge for edge in self.edge_adjacency[opposite] if node not in edge]
            for edge in self.edges:
                if node in edge:
                    self.edges.remove(edge)
        else:
            raise ValueError("Node not in Network")
        
    def remove_nodes(self, nodes):
        for node in nodes:
            self.remove_node(node)
            
    def remove_edge(self, edge):
        if edge in self.edges:
            self.edges.remove(edge)
            self.adjacency[edge[0]].remove(edge[1])
            self.edge_adjacency[edge[0]].remove(edge)
            if not edge.dir:
                self.adjacency[edge[1]].remove(edge[0])
                self.edge_adjacency[edge[1]].remove(edge)
        else:
            raise ValueError("Edge not in Network")
        
    def remove_edges(self, edges):
        for edge in edges:
            self.remove_edge(edge)
    
    #############################################
    # Components
    #############################################
    
    def maximal_node_component(self,node):
        visited = set()
        queue = deque([node])
        while queue:
            next = queue.popleft()
            visited.add(next)
            queue.extend([i for i in self.adjacency[next] if i not in visited])
        return visited
    
    def maximal_subnetwork(self,node):
        nodes = deque(self.maximal_node_component(node))
        nodesres = nodes.copy()
        edges = []
        while nodes:
            next = nodes.popleft()
            edges.extend(self.edge_adjacency[next])
        edges = list(set(edges))
        return Network(nodesres,edges)
        
    def maximal_node_subnetworks(self):
        to_visit = set(self.nodes.copy())
        node_networks = []
        while to_visit:
            start = to_visit.pop()
            node_network = self.maximal_node_component(start)
            to_visit.difference_update(node_network)
            node_networks.append(node_network)
        return node_networks
    
    def links_between(self, subnet1, subnet2):
        links = []
        for node1,node2 in product(subnet1.nodes,subnet2.nodes):
            links.extend(self.get_edges(node1,node2))
        return links
    
    def remove_all_edges(self):
        self.edges = []
        for node in self.nodes:
            self.adjacency[node] = []
            self.edge_adjacency[node] = []
        print('success')
        
    def max_key(self, node1, node2):
        return max(self.func_edge_compare(lambda x : x.weight , node1, node2, reduce = False))
        
    def create_edge(self, node1, node2, dir = False, weight = 1):
        if node1 not in self.nodes or node2 not in self.nodes:
            raise ValueError("Node not in Network")
        elif not self.get_edges(node1,node2):
            self.add_edge(Edge(node1, node2, dir, weight))
        else:
            key = self.max_key(node1, node2)
            self.add_edge(Edge(node1, node2, dir = dir, weight = weight, key = key + 1))

    def complete_network(self):
        return Network(self.nodes, [Edge(node1, node2, dir = False, weight = 1) for node1, node2 in product(self.nodes, self.nodes)])

    def is_isolated(self, node) -> bool:
        return bool(self.adjacency[node])
    
    def has_loop(self, node) -> bool:
        return node in self.adjacency[node]
    
    def is_connected(self) -> bool:
        return len(self.maximal_node_subnetworks()) == 1
    
    def is_complete(self) -> bool:
        return all(self.degree(node) == len(self.nodes) - 1 for node in self.nodes)
    
    def degree(self, node) -> int:
        return len(self.adjacency[node])
    
    #############################################
    # Paths
    #############################################
    
    def is_path(self, nodes):
        return all(nodes[i+1] in self.adjacency[nodes[i]] for i in range(len(nodes)-1))
    
    def find_path(self,origin,goal):
        queue = deque()
        path = []
        visited = {origin}
        queue.extend([(i,origin) for i in self.adjacency[origin] if i not in visited])
        
        while queue:
            next = queue.popleft()
            path.append(next)
            if next[0] == goal:
                def resolve_path(path):
                    goal = path[0][1]
                    step = path[-1][0]
                    rev = path[::-1]
                    steps = list(map(lambda x: x[0], rev))
                    origins = list(map(lambda x: x[1], rev))
                    new = [step]
                    while step != goal:
                        step = origins[steps.index(step)]
                        new.append(step)
                    return new[::-1]
                return resolve_path(path)
            visited.add(next[0])
            queue.extend([(i,next[0]) for i in self.adjacency[next[0]] if i not in visited])
        return []
    
    def lightest_edges_of_path(self, node1, node2):
        path = self.find_path(node1, node2)
        for i in range(len(path)-1):
            yield reduce(lambda x ,y : x.weight if x.weight < y.weight else y.weight,[edge for edge in self.edge_adjacency[path[i]] if path[i+1] in edge])
   
    def path_weight(self, path):
        return sum(self.get_edge(path[i], path[i+1]).weight for i in range(len(path)-1))
   
    def find_path_with_func(self, origin, goal, func):
        queue = deque()
        path = []
        visited = {origin}
        queue.extend([(i,origin) for i in self.adjacency[origin] if i not in visited])
        
        while queue:
            next = queue.popleft()
            path.append(next)
            if next[0] == goal:
                def resolve_path(path):
                    goal = path[0][1]
                    step = path[-1][0]
                    rev = path[::-1]
                    steps = list(map(lambda x: x[0], rev))
                    origins = list(map(lambda x: x[1], rev))
                    new = [step]
                    while step != goal:
                        step = origins[steps.index(step)]
                        new.append(step)
                    return new[::-1]
                return resolve_path(path)
            visited.add(next[0])
            queue.extend([(i,next[0]) for i in self.adjacency[next[0]] if i not in visited])
        return []
    
    class Path:
        def __init__(self, network, path):
            self.network = network
            self.path = path
            self.weight = self.network.path_weight(path)
        
        def __repr__(self):
            return " -> ".join([str(node) for node in self.path])
                
        def __str__(self):
            return " -> ".join([str(node) for node in self.path])
        
        def __len__(self):
            return len(self.path)
        
        def __getitem__(self, index):
            return self.path[index]
        
        def __iter__(self):
            return iter(self.path)
        
        def __reversed__(self):
            return reversed(self.path)
        
        def __contains__(self, item):
            return item in self.path
        
        def __eq__(self, other):
            if self.path[0] == other.path[0] and self.path[-1] == other.path[-1]:
                return self.weight == other.weight
            else:
                raise ValueError("Paths do not share the same start and end nodes")
        
        def __ne__(self, other):
            return self.path != other.path
        
        def __lt__(self, other):
            if self.path[0] == other.path[0] and self.path[-1] == other.path[-1]:
                return self.weight < other.weight
            else:
                raise ValueError("Paths do not share the same start and end nodes")
            
        def __le__(self, other):
            if self.path[0] == other.path[0] and self.path[-1] == other.path[-1]:
                return self.weight <= other.weight
            else:
                raise ValueError("Paths do not share the same start and end nodes")
            
        def __gt__(self, other):
            if self.path[0] == other.path[0] and self.path[-1] == other.path[-1]:
                return self.weight > other.weight
            else:
                raise ValueError("Paths do not share the same start and end nodes")
            
        def __ge__(self, other):
            if self.path[0] == other.path[0] and self.path[-1] == other.path[-1]:
                return self.weight >= other.weight
            else:
                raise ValueError("Paths do not share the same start and end nodes")
            
        def __hash__(self):
            return hash(self.path)
        
        def __add__(self, other):
            if self.path[-1] == other.path[0]:
                return Network.Path(self.network, self.path + other.path[1:])
            else:
                raise ValueError("Path do not meet at the same node")
        
        @property
        def start(self):
            return self.path[0]
        
        @property
        def end(self):
            return self.path[-1]
        
        @property
        def nodes(self):
            return self.path
        
        
        def extremes(self):
            return self.path[0], self.path[-1]
        
        def possible_siblings(self):
            siblings = {}
            for index,node in enumerate(self.path[:-1]):
                siblings[node] = self.network.get_edges(node, self.path[index+1])
            return siblings
        
        def select_sibling(self , func ):
            siblings = self.possible_siblings()
            edges = []

            for node in self.path:
                edges.extend(reduce(func , siblings[node]))
            return edges
                
        def lightest_sibling(self):
            return self.select_sibling(lambda x,y: x if x.weight < y.weight else y)
        
        def contains_cycle(self):
            return self.path[0] in self.path[1:]
        
        def remove_cycles(self):
            if self.contains_cycle():
                return Network.Path(self.network, self.path[:self.path.index(self.path[0])+1]).remove_cycles()
            else:
                return self
        
