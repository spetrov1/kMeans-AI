import numpy as np
import matplotlib.pyplot as plt
from typing import List
import re
import random
import math

def extract_file_lines(filePath: str) -> List[str]:
    file1 = open(filePath, 'r') 
    lines = file1.readlines()

    return lines

def parse_file_line_to_point(line: str):
    line = line.replace('\t', ' ')
    line = line.replace('\n', '')
    coords = line.split(' ')

    x = coords[0]
    y = coords[1]

    return (x, y)

def extract_points_given_file(filePath: str):
    lines = extract_file_lines(filePath)
    points = []

    for line in lines:
        point = parse_file_line_to_point(line)
        points.append(point)

    return points

# euclidean distance
def dist(point1: (float, float), point2: (float, float)):
    return math.sqrt( ((point1[0]-point2[0])**2)+((point1[1]-point2[1])**2) )


class Cluster:
    
    def __init__(self):
        self.points = []
        self.centroid = None

    def compute_centroid(self) -> None:

        if len(self.points) == 0:
            self.centroid = None
            return

        x = 0
        y = 0
        pointsCount = len(self.points)

        for point in self.points:
            x = x + point[0]
            y = y + point[1]

        self.centroid = (x / pointsCount, y / pointsCount)

    def add_point(self, point):
        self.points.append(point)

    def get_centroid(self) -> (float, float):
        return self.centroid

    # TODO Check for correctness
    def compute_internal_cluster_distance(self):
        # clusterSize = len(self.points)
        internalClusterDistance = 0

        for point in self.points:
            internalClusterDistance += pow(dist(point, self.centroid), 2) 
        
        # internalClusterDistance = internalClusterDistance * clusterSize
        return internalClusterDistance

    def empty_cluster(self):
        self.points = []


# TODO try to random init points but not from given points
def random_init_clusters(points: (float, float), numCentroids: int) -> List[Cluster]:
    centroids = random.sample(points, numCentroids)
    clusters = []

    for centroid in centroids:
        cluster = Cluster()
        cluster.add_point(centroid)

        clusters.append(cluster)

    return clusters

def assign_point_to_nearest_cluster(point, clusters: List[Cluster]):
    distances = [dist(point, cluster.get_centroid()) for cluster in clusters]
    clusterIndex = distances.index(min(distances))

    clusters[clusterIndex].add_point(clusterIndex)

# step 1 of kMeans in cycle
def assign_points_to_nearest_clusters(points, clusters: List[Cluster]):
    for point in points:
        assign_point_to_nearest_cluster(point, clusters)

# step 2 of kMeans in cycle
# compute all the centroids
def update_centroids(clusters: List[Cluster]):
    for cluster in clusters:
        cluster.compute_centroid()

def empty_clusters(clusters: List[Cluster]):
    for cluster in clusters:
        cluster.empty_cluster()


# def kMeans(k, filePathWithData) -> List[Cluster]:
#     points = extract_points_given_file(filePathWithData)
#     clusters = random_init_clusters(points, k)

#     while centroid_differs(lastCentroids, currentCentroids):
#         update_centroids(clusters)
#         empty_clusters(clusters)
#         assign_points_to_nearest_clusters(points, clusters)


print( (0, 1) in ( (0, 0), (1, 1) ) )

# cl = Cluster()
# cl.add_point((1, 0))

# clusters = [cl]

# cl.compute_centroid()
# print(cl.get_centroid())

# empty_clusters(clusters)
# update_centroids(clusters)

# print(cl.get_centroid())


exit()



# points = extract_points_given_file('normal/normal.txt')
# print(points[0])

# # exit()

# # N = 3
# # x = np.random.rand(N)
# # y = np.random.rand(N)

# # plt.scatter(x, y, color='black')
# # plt.show()

# x = [1, 2, 3]
# y = [1, 2, 3]

# plt.scatter(x, y, color='red')

# plt.scatter([10, 20], [10, 20], color='black')

# plt.show()