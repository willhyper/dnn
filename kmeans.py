# Example with a richer dataset.
# See: https://www.datascience.com/blog/introduction-to-k-means-clustering-algorithm-learn-data-science-tutorials

from pprint import pprint
from math import fsum, sqrt
from collections import defaultdict
from functools import partial
from random import sample

def dist(p,q):
    return sqrt(fsum( (x1-x2)**2 for x1, x2 in zip(p,q) ) )

def mean(data):
    data = list(data) # because people expect generator too
    return fsum(data)/len(data)

def transpose(m):
    return zip(*m)

def assign_data(centroids, points):
    d = defaultdict(list)
    for p in points:
    	c = min(centroids, key = partial(dist, p))
        # print(p,'is closest to',c)
    	d[c].append(p)
    return dict(d) # why not just return defaultdict? people knows dict better than defaultdict

def compute_centroids(groups):
    return [tuple(map(mean, transpose(g))) for g in groups]


def kmeans(data, k, iterations=100): # with assumption that centroids converge
    centroids = sample(data,k)
    for _ in range(iterations):
        labeled = assign_data(centroids, data)
        centroids = compute_centroids(labeled.values())
    return centroids

if __name__ == '__main__':

    print('Simple example with six 3-D points clustered into two groups')
    
    points = [
            (10, 41, 23),
            (22, 30, 29),
            (11, 42, 5),
            (20, 32, 4),
            (12, 40, 12),
            (21, 36, 23),
            ]

    centroids = kmeans(points, k=2)
    
    pprint(points)
    print('centroids',centroids)

    # data = [
    #          (10, 30),
    #          (12, 50),
    #          (14, 70),
    #
    #          (9, 150),
    #          (20, 175),
    #          (8, 200),
    #          (14, 240),
    #
    #          (50, 35),
    #          (40, 50),
    #          (45, 60),
    #          (55, 45),
    #
    #          (60, 130),
    #          (60, 220),
    #          (70, 150),
    #          (60, 190),
    #          (90, 160),
    #         ]
