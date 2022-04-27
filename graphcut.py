## IMPLEMENTATION OF KOLMOGOROV ZABIH GRAPH CUT METHOD
## https://www.ipol.im/pub/art/2014/97/

import numpy as np
import cv2
import os
import maxflow

## FUNCTION HEADERS AND VARIABLE DEFINITIONS FROM THE KOLMOGOROV-ZABIH PAPER
## https://www.ipol.im/pub/art/2014/97/

OCCLUDED = 2**28
ALPHA = -5
ABSENT = -6
K = 15
LAMBDA = 3

def ssd(img1, img2, p1, p2):
    return abs(img1[p1] - img2[p2])

def forbid01(g, n1, n2): 
    g.add_edge(n1, n2, OCCLUDED, 0)

def is_var(var): 
    return (var != ALPHA) and (var != ABSENT)

def is_valid(p1, shape):
    return (0 <= p1[0] < shape[0]) and (0 <= p1[1] < shape[1])

def add_pair(g, n1, n2, a, b, c, d):
    g.add_tedge(n1, d, b)
    g.add_tedge(n2, 0, a-b)
    g.add_edge(n1, n2, 0, (b + c)-(a + d))

class Graphcut:
    def __init__(self, img1, img2):
        self.img1, self.img2 = img1, img2
        self.labels = np.ones(self.img1.shape, dtype=np.int) * OCCLUDED
        self.varsA = np.zeros(self.img1.shape, dtype=np.int)
        self.vars0 = np.zeros(self.img1.shape, dtype=np.int)
        self.energy = np.inf

    def get_lambda(self, p1, p2, disp):
        a = abs(self.img1[p1] - self.img1[p2])
        b = abs(self.img2[p1[0], p1[1] + disp] - self.img2[p2[0], p2[1] + disp])
        lamb = 3*LAMBDA if (max(a, b) < 8) else LAMBDA
        return lamb
    
    def run(self, iters, mindisp, maxdisp):
        for _ in range(iters):
            labels = np.random.permutation(range(mindisp, maxdisp))
            for label in labels:
                g = self.build_graph(label) ## build graph, add data terms
                self.add_terms(g, label) ## add smoothness and uniqueness
                maxflow = g.maxflow()
                new_energy = maxflow + self.penalty
                print(f"current energy: {self.energy}, new energy: {new_energy}")
                if new_energy < self.energy:
                    self.energy = new_energy
                    self.update_disparity(g, label)
        return self.labels
    
    def build_graph(self, label):
        self.penalty = 0
        npixels = self.img1.shape[0] * self.img1.shape[1]
        g = maxflow.Graph[int](2*npixels, 12*npixels)
        for r in range(self.img1.shape[0]):
            for c in range(self.img1.shape[1]):
                disp = self.labels[r, c]
                p1, p2 = (r, c), (r, c+disp)
                if disp == label: ## not occluded
                    self.varsA[p1] = ALPHA
                    self.vars0[p1] = ALPHA
                    pen = ssd(self.img1, self.img2, p1, p2) - K
                    self.penalty += pen
                else:
                    ## vars0, if occluded, then mark absent, else create node
                    if disp == OCCLUDED:
                        self.vars0[p1] = ABSENT
                    else:
                        self.vars0[p1] = g.add_nodes(1)[0]
                        pen = ssd(self.img1, self.img2, p1, p2) - K
                        g.add_tedge(self.vars0[p1], 0, pen)
                    
                    ## varsA, if in right image, , then create node, else mark absent
                    p2 = (r, c + label)
                    if is_valid(p2, self.img2.shape):
                        self.varsA[p1] = g.add_nodes(1)[0]
                        pen = ssd(self.img1, self.img2, p1, p2) - K
                        g.add_tedge(self.varsA[p1], pen, 0)
                    else:
                        self.varsA[p1] = ABSENT
        return g

    def add_terms(self, g, label):
        for r in range(self.img1.shape[0]):
            for c in range(self.img1.shape[1]):
                p1 = (r, c)
                ## smoothness
                for p2 in [(r, c-1), (r+1, c), (r, c+1), (r-1, c)]:
                    if is_valid(p2, self.img1.shape):
                        a1, a2 = self.varsA[p1], self.varsA[p2]
                        o1, o2 = self.vars0[p1], self.vars0[p2]
                        d1, d2 = self.labels[p1], self.labels[p2]
                        if is_var(a1) and is_var(a2):
                            pen = self.get_lambda(p1, p2, label)
                            add_pair(g, a1, a2, 0, pen, pen, 0)
                        elif is_var(a1) and (a2 == ALPHA):
                            pen = self.get_lambda(p1, p2, label)
                            g.add_tedge(a1, 0, pen)
                        elif is_var(a2) and (a1 == ALPHA):
                            pen = self.get_lambda(p1, p2, label)
                            g.add_tedge(a2, 0, pen)
                                                
                        v3 = (p2[0], p2[1] + d1)
                        v4 = (p1[0], p1[1] + d2)
                        if (d1 == d2) and is_var(o1) and is_var(o2):
                            pen = self.get_lambda(p1, p2, d1)
                            add_pair(g, o1, o2, 0, pen, pen, 0)
                        elif (d1 != d2) and is_var(o1) and is_valid(v3, self.img1.shape):
                            pen = self.get_lambda(p1, p2, d1)
                            g.add_tedge(o1, 0, pen)
                        elif (d1 != d2) and is_var(o2) and is_valid(v4, self.img1.shape):
                            pen = self.get_lambda(p1, p2, d2)
                            g.add_tedge(o2, 0, pen)
                
                ## uniqueness
                disp = self.labels[r, c]
                p1 = (r, c)
                a, o = self.varsA[p1], self.vars0[p1]
                if is_var(o) and (a != ABSENT):
                    forbid01(g, o, a)
                
                p2 = (r, c - label + int(disp))
                if is_valid(p2, self.img1.shape):
                    forbid01(g, o, self.varsA[p2])

    def update_disparity(self, g, label):
        for r in range(self.labels.shape[0]):
            for c in range(self.labels.shape[1]):
                a, o = self.varsA[r, c], self.vars0[r, c]
                if (is_var(o)) and (g.get_segment(o) == 1):
                    self.labels[r,c] = OCCLUDED
                if (is_var(a)) and (g.get_segment(a) == 1):
                    self.labels[r,c] = label