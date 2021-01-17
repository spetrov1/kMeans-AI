import numpy as np
import sys
import matplotlib.pyplot as plt
from typing import List
import re
import random
import math
import copy 

def extract_file_lines(filePath: str) -> List[str]:
    file1 = open(filePath, 'r') 
    lines = file1.readlines()

    return lines

def parse_file_line_to_point(line: str):
    line = line.replace('\t', ' ')
    line = line.replace('\n', '')
    coords = line.split(' ')

    x = int(coords[0])
    y = int(coords[1])

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
    
    def __init__(self, centroid: (float, float)):
        # first point in self.points is the init centroid
        self.points = [centroid]
        self.centroid = centroid

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

    # remove all points from cluster
    #  except the first one which is the init centroid
    def empty_cluster(self):
        self.points = self.points[0:1]

    def get_points_first_coord(self):
        return [point[0] for point in self.points]

    def get_points_second_coord(self):
        return [point[1] for point in self.points]

# TODO try to random init points but not from given points
def random_init_clusters(points: (float, float), numCentroids: int) -> List[Cluster]:
    centroids = random.sample(points, numCentroids) # TODO does this get different points
    # points = [point for point in points if point not in centroids]

    # TODO try to init random points (but with int coordinates)
    # centroids = []
    # for _ in range(numCentroids):
    #     centroids.append( (random.randint(3, 1009), random.randint(3, 1009)) )

    clusters = []

    for centroid in centroids:
        cluster = Cluster(centroid)
        clusters.append(cluster)

    return clusters

def get_nearest_center(centroids: (float, float), point) -> (float, float):
    distances = [dist(centroid, point) for centroid in centroids]
    index = distances.index(min(distances))

    return centroids[index]

def assign_point_to_nearest_cluster(point, clusters: List[Cluster]):
    distances = [dist(point, cluster.get_centroid()) for cluster in clusters]
    clusterIndex = distances.index(min(distances))

    clusters[clusterIndex].add_point(point)

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

def get_centroids(clusters: List[Cluster]):
    centroids = [cluster.get_centroid() for cluster in clusters]
    return centroids

def k_means_given_file(k, filePathWithData) -> List[Cluster]:
    points = extract_points_given_file(filePathWithData)
    return k_means(k, points)

def k_means(k, points):
    clusters = random_init_clusters(points, k)
    # update_centroids(clusters)

    lastCentroids = None
    currentCentroids = get_centroids(clusters)
    def centroids_differs(lastCentroids, currentCentroids):
        return not (lastCentroids == currentCentroids)

    while centroids_differs(lastCentroids, currentCentroids):
        lastCentroids = copy.deepcopy(currentCentroids)
        empty_clusters(clusters)
        assign_points_to_nearest_clusters(points, clusters)
        update_centroids(clusters)
        currentCentroids = get_centroids(clusters)

    return clusters  

import matplotlib.colors as mcolors
def displayClusters(clusters: List[Cluster]):
    def spreadPoints(cluster: Cluster, colorName: str):
        xCoords = cluster.get_points_first_coord()
        yCoords = cluster.get_points_second_coord()
        plt.scatter(xCoords, yCoords, color=colorName)
    
    # colors = list(mcolors.CSS4_COLORS.values())
    colors = list(mcolors.TABLEAU_COLORS.values())
    

    for index, cluster in enumerate(clusters):
        spreadPoints(cluster, colors[index])

    plt.show()

# evaluate clusters by their internal distance
# the smaller evaluation means the better solution
def evaluate(clusters: List[Cluster]) -> float:
    clustersEvaluation = 0

    for cluster in clusters:
        clustersEvaluation += cluster.compute_internal_cluster_distance()
    
    return clustersEvaluation

# apply k means algorithm multiple times and get the best solution
def apply_k_means_multiple_times(k: int, filePathWithData: str, nTimes) -> List[Cluster]:
    currentBestValue = sys.maxsize
    currentSolutionClusters = []
    points = extract_points_given_file(filePath)

    while nTimes > 0:
        clusters = k_means(k, points)
        value = evaluate(clusters)
        if value < currentBestValue:
            currentBestValue = value
            currentSolutionClusters = clusters
        nTimes = nTimes - 1
    
    return currentSolutionClusters


# get random value from @values
# values[index] has chance chances[index]
def get_random_with_chances(values: List, chances: List[float]):
    allChances = 0
    for chance in chances:
        allChances += chance
    
    randomChance = allChances * random.random()

    sumChances = 0
    for index, chance in enumerate(chances):
        if index < len(chances) and sumChances + chances[index] >= randomChance:
            return values[index]
        sumChances += chance
    return None

def k_means_plus_plus_initialization(points, k):
    # TODO when points is chosen then remove from points
    centroids = random.sample(points, 1)
    probabilities = []

    while len(centroids) != k:
        for point in points:
            nearestCenter = get_nearest_center(centroids, point)
            distance = dist(nearestCenter, point)**2
            probabilities.append(distance)
        chosenPoint = get_random_with_chances(points, probabilities)
        centroids.append(chosenPoint)
        probabilities = []
    return centroids



filePath = 'C:\\Users\\35988\\Desktop\\HW7\\unbalance\\unbalance.txt'
points = extract_points_given_file(filePath)

for _ in range(10):
    centroids = k_means_plus_plus_initialization(points, 8)
    clusters = []
    for centroid in centroids:
        clusters.append(Cluster(centroid))
    displayClusters(clusters)

exit()



for _ in range(10):
    random2 = get_random_with_chances([1, 2, 3], [0, 57, 2])
    print(random2)
exit()

filePath = 'C:\\Users\\35988\\Desktop\\HW7\\unbalance\\unbalance.txt'
numClusters = 8
clusters = apply_k_means_multiple_times(numClusters, filePath, 5)
displayClusters(clusters)


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