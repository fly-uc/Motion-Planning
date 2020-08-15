#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from queue import PriorityQueue

plt.rcParams['figure.figsize'] = 12, 12

def city_map():
    grid = np.zeros([100,100])
    grid[10:20, 10:20] = 1
    grid[63:80, 10:20] = 1
    grid[43:60, 30:40] = 1
    grid[71:86, 38:50] = 1
    grid[10:20, 55:67] = 1
    grid[80:90, 80:90] = 1
    grid[75:90, 80:90] = 1
    grid[30:40, 60:82] = 1
    return grid
    
def random_vertex(grid):
    x = np.random.uniform(0, grid.shape[0])
    y = np.random.uniform(0, grid.shape[1])
    return (x,y)

def nearest_neighbour(q_rand, rrt):
    nearest_distance = float("inf")
    nearest_vertex = None
    for v in rrt.vertices:
        euclidean_distance = np.linalg.norm(np.array(q_rand)-np.array(v[:2]))
        if euclidean_distance < nearest_distance: 
            nearest_distance = euclidean_distance
            nearest_vertex = v
    return nearest_vertex
    
def orientation(q_rand, q_near):
    orient = np.arctan2(q_rand[1]-q_near[1], q_rand[0]-q_near[0])
    return orient

def new_vertex(q_near, direct, dt):
    x2 = q_near[0] + np.cos(direct)*dt 
    y2 = q_near[1] + np.sin(direct)*dt
    return (x2,y2)

class RRT:
    def __init__(self, q_init):
        self.tree = nx.Graph()
        self.tree.add_node(q_init)
        
    def add_vertex(self, q_new):
        self.tree.add_node(tuple(q_new))
        
    def add_edge(self, q_near, q_new, u):
        self.tree.add_edge(tuple(q_near), tuple(q_new), orientation = u)
            
    def edge_cost(self, current_node, next_node):
        self.tree.edges[current_node, next_node]['weight']
            
    @property
    def vertices(self):
        return self.tree.nodes()
    
    @property
    def edges(self):
        return self.tree.edges()
            
def build(grid, q_init, number_of_vertices, dt):
    
    rrt = RRT(q_init)
    
    for i in range (number_of_vertices):
        q_rand = random_vertex(grid)
        
        while grid[int(q_rand[0]), int(q_rand[1])] == 1:
            q_rand = random_vertex(grid)
            
        q_near = nearest_neighbour(q_rand,rrt)
        orient = orientation(q_rand, q_near)
        q_new  = new_vertex(q_near, orient, dt)
        
        if grid[int(q_new[0]),int(q_new[1])] == 0:
              
            rrt.add_edge(q_near, q_new, orient)
                   
    return rrt

def heuristic(n1, n2):
    return np.linalg.norm(np.array(n2) - np.array(n1))

def a_star(graph, h, start, goal):
           
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
     
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                branch_cost = current_cost + cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node)
                    queue.put((queue_cost, next_node))
                    
    if found:
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('Failed to find a path!')
    return path[::-1], path_cost

number_of_vertices = 1500
dt = 2
# q_init = (50, 50)
start_ne = (20, 90)
goal_ne = (80, 30)
q_init = start_ne

grid = city_map()
rrt = build(grid, q_init, number_of_vertices, dt)

plt.imshow(grid, cmap='Greys', origin='lower')
# plt.plot(q_init[1], q_init[0], 'go')

for (v1, v2) in rrt.edges:
    plt.plot([v1[1], v2[1]], [v1[0], v2[0]], 'r-')

start_ne_g = nearest_neighbour(start_ne, rrt)
goal_ne_g = nearest_neighbour(goal_ne, rrt)
print(start_ne_g)
print(goal_ne_g)

G = nx.Graph()
for e in rrt.edges:
    p1 = e[0]
    p2 = e[1]
    dist = np.linalg.norm(np.array(p2) - np.array(p1))
    G.add_edge(p1, p2, weight=dist)

path, cost = a_star(G, heuristic, start_ne_g, goal_ne_g)
print(len(path)) 

plt.plot([start_ne[1], start_ne_g[1]], [start_ne[0], start_ne_g[0]], 'b-')
for i in range(len(path)-1):
    p1 = path[i]
    p2 = path[i+1]
    plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'b-')
plt.plot([goal_ne[1], goal_ne_g[1]], [goal_ne[0], goal_ne_g[0]], 'b-')
    
plt.plot(start_ne[1], start_ne[0], 'gx')
plt.plot(goal_ne[1], goal_ne[0], 'gx')

plt.xlabel('EAST', fontsize=20)
plt.ylabel('NORTH', fontsize=20)
plt.show()
