import scipy.sparse as ssp
import numpy as np
import time
import random
np.random.seed(42)
def time_profile(fun,name,*args):
    times = []
    for i in range(10):    
        st = time.time()
        r = fun(args[0])
        et = time.time()
        times.append(et-st)
    print("Time taken for ", name,": ", np.average(times))
    return r

def neighbors (fringe,A):
    return set(A[list(fringe)].indices)
    
## Resizing CSR
def rts1(r:ssp.csr_matrix):
    num = r.shape[1]
    sT = r.transpose()
    sT.resize((num,num))
    sT[0] = r[0]
    sT[1] = r[1]
    return sT

## Numpy Zeroes
def rts2(r:ssp.csr_matrix):
    num = r.shape[1]
    arr = r.toarray()
    tmp = np.zeros((num,num))
    tmp[0] = arr[0]
    tmp[1] = arr[1]
    sT = tmp.transpose()
    sT[0] = arr[0]
    sT[1] = arr[1]
    return ssp.csr_matrix(sT)

## Numpy Transpose
def rts3(r:ssp.csr_matrix):
    num = r.shape[1]
    r.resize((num,num))
    arr = r.toarray()
    arrT = arr.transpose()
    arrT[0] = arr[0]
    arrT[1] = arr[1]
    return ssp.csr_matrix(arrT)

## Lil Matrix
def rts4(r:ssp.csr_matrix):
    num = r.shape[1]
    lil = ssp.lil_matrix((num,num))
    lil[0] = r[0]
    lil[1] = r[1]
    lilT = lil.transpose()
    lilT[0] = r[0]
    lilT[1] = r[1]
    return lilT.tocsr()

## DOK Matrix
def rts5(r:ssp.csr_matrix):
    num = r.shape[1]
    dok = ssp.dok_matrix((num,num))
    dok[0] = r[0]
    dok[1] = r[1]
    dokT = dok.transpose()
    dokT[0] = r[0]
    dokT[1] = r[1]
    return dokT.tocsr()

## CSC Matrix
def rts6(r:ssp.csr_matrix):
    num = r.shape[1]
    c = r.tocsc().transpose()
    c.resize((num,num))
    c[0] = r[0]
    c[1] = r[1]
    return c

def get_subgraph(src,dst,A):
    st = time.time()
    nodes = [src,dst]
    dists = [0,0]
    fringe = neighbors(nodes,A)
    fringe = fringe - set(nodes)

    ## Construction
    nodes = nodes + list(fringe)
    dists = dists + [1] * len(fringe)
    
    print("Baseline All Nodes: ", nodes)
    subgraph = A[[src,dst],:][:,nodes]
    print("Baseline subgraph: ", subgraph)
    subgraph[0,1]=0
    subgraph[1,0]=0
    et = time.time()
    print("Initial Overhead Cost: ", et-st)
    args = subgraph
    funcs = [rts6]
    subs = []
    i = 0
    for func in funcs:
        i += 1
        name = f"rts{i}" 
        subs.append(time_profile(func,name,args))

    return subs

def get_1_hop_subgraph(src,dst,A,k,max_nodes=None,node_features=None,ratio=1.0,y=1):
    # st = time.time()
    dists = []
    visited = [src, dst]
    fringe = set([src, dst])
    nodes = [src,dst]
    dists = [0,0]
    fringe = neighbors(fringe, A)
    if ratio < 1.0:
        fringe = set(random.sample(fringe, int(ratio*len(fringe))))
    if max_nodes is not None:
        if max_nodes < len(fringe):
            fringe = set(random.sample(fringe, max_nodes))
    fringe = fringe - set(visited)
    visited = visited + list(fringe)

    print("Included Nodes: ", nodes)
    print("All Nodes: ", visited)
    subgraph = A[list(nodes), :][:, list(visited)]
    print("New subgraph: ", subgraph)
    
    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0
    transformed = rts6(subgraph)
    
    if node_features is not None:
        node_features = node_features[nodes]

    return transformed

def get_k_hop_subgraph(src,dst,A,k,max_nodes=None,node_features=None,ratio=1.0,y=1):
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, k+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if ratio < 1.0:
            fringe = random.sample(fringe, int(ratio*len(fringe)))
        if max_nodes is not None:
            if max_nodes < len(fringe):
                fringe = random.sample(fringe, max_nodes)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
        
    subgraph = A[list(visited), :][:, nodes]
    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0
    transformed = rts6(subgraph)
    
    if node_features is not None:
        node_features = node_features[nodes]
        
    return transformed


mat = [
    [0,1,1,0,1],
    [1,0,1,0,0],
    [1,1,0,1,0],
    [0,0,1,0,0],
    [1,0,0,0,0]
       ]

big_mat = [
    0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 
    1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
    1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 
    0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
    0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 
    0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 
]

big_mat = np.array(big_mat).reshape((13,13))
# A = ssp.csr_matrix(mat, shape=(5,5))
A = ssp.csr_matrix(big_mat)
print("\nA CSR")
print(A.toarray())
# print(A)
# print(f"src =\n{A[0]}\n")
# print(f"dst =\n{A[1]}\n")
src = 2
dst = 4

# subs = get_subgraph(src,dst,A)
subs = [get_1_hop_subgraph(src,dst,A,1),get_k_hop_subgraph(src,dst,A,1)]
subs = np.concatenate([subs, get_subgraph(src,dst,A)])
print("\nPrinting subgraphs")
print("-"*30)
i = 1
for sub in subs:
    print("\nSubgraph from rts", i)
    print(sub.toarray())
    i += 1
    
print("-"*30)

# multiplier = 1 / np.log(A.sum(axis=0))
# multiplier[np.isinf(multiplier)] = 0
# print(f"Multiplier=\n{multiplier}\n")
# A_ = ssp.csr_matrix(A.multiply(multiplier))
# print(f"A_=\n{A_}\n")
# # A_csr = A_.tocsr()
# # print(f"A_csr =\n{A_csr}\n")

# # print(f"A[src] = \n{A[src]}\n")
# # print(f"A_[dst] = \n{A_csr[dst]}\n")
# AA_scores = A[src].multiply(A_[dst])
# print(f"AA_scores = \n{AA_scores}\n")
# print(f"AA_scores array = \n {AA_scores.toarray()}\n")

# pruned = np.argsort(AA_scores.toarray()).flatten()
# print(f"Sorted scores = \n{pruned}\n")
# # p = .2
# # final = pruned[-int(len(pruned)*p):]
# n = 3
# final = pruned[-int(n):]
# print(f"final = {final}")
# print(f"Neighbors = \n{A[list(final)]}")
# print(f"Neighbor Indices: {A[list(final)].indices}")



