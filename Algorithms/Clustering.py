import matplotlib.pyplot as plt
import numpy as np
import random


colors = ["g", "r", "c", "b", "k"]


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=200):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            if i == self.max_iter-2:
                print("Max iteration reached, terminating")
            self.distinct_groups = {}

            for i in range(self.k):
                self.distinct_groups[i] = []

            for feature in data:
                distances = []
                for centroid in self.centroids:
                    distances.append(np.linalg.norm(feature - self.centroids[centroid]))   # Euclidean distance
                closest_centroid = distances.index(min(distances))
                # Build a collection of features for each centroid
                self.distinct_groups[closest_centroid].append(feature)
            prev_centroids = self.centroids.copy()                             # Needed to calculate centroid movement
            for group in self.distinct_groups:
                self.centroids[group] = np.average(self.distinct_groups[group], axis=0)
            opt = True
            for centroid in self.centroids:
                old_center = prev_centroids[centroid]
                current_center = self.centroids[centroid]
                percent_change = abs(np.sum((current_center - old_center)/old_center*100))
                if percent_change > self.tol:
                    print("Centroid " + str(centroid) + ": " +str(percent_change))
                    opt = False
            if opt:
                break
    def predict(self, data):
        distances = []
        for centroid in self.centroids:
            distances.append(np.linalg.norm(data - self.centroids[centroid]))  # Euclidean distance
        closest_centroid = distances.index(min(distances))
        return closest_centroid


class MeanShift:
    def __init__(self, radius):
        self.radius = radius

def generate_data(iteration, axes = 2, bias = .04, bias_freq = .5):
    temp = []

    for i in range(iteration):
        if random.random() < bias_freq:
            factor = bias*100
        else:
            factor = 100
        x = int(random.random()*factor)
        y = int(random.random()*factor)
        temp.append([x,y])
    data = np.array(temp)
    return data


iteration = int(input("Enter Data Amount: "))

total_clusters = int(input("Enter Cluster Amount: "))
data = generate_data(iteration)
clf = K_Means(total_clusters)
clf.fit(data)

# Visualization
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=50, linewidths=2)
for group in clf.distinct_groups:
    color = colors[group % 5]
    for feature in clf.distinct_groups[group]:
        plt.scatter(feature[0], feature[1], marker="x", color=color, s=50, linewidths=2)

# Classify unknown data into existing clusters
unknowns = generate_data(10)
for unknown in unknowns:
    classification = clf.predict(unknown)
    color = colors[classification % 5]
    plt.scatter(unknown[0], unknown[1], marker="*", color=color, s=25, linewidths=5)

plt.show()