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
    def __init__(self, radius=4, step = 100):
        self.radius = radius
        self.step = step
    def fit(self, data):
        centers = {}
        for i in range(len(data)):
            # Make every data point a center
            centers[i] = data[i]
        while True:
            new_centers = []
            for i in centers:
                center = centers[i]
                within_radius = []
                for point in data:
                    if np.linalg.norm(point-center) < self.radius:
                        within_radius.append(point)
                # Make new center the mean of features within the radius (bandwidth)
                new_centers.append(tuple(np.average(within_radius, axis=0)))

            unique_centers = sorted(list(set(new_centers)))     # Combine centers with the same location
            previous_centers = centers.copy()
            centers.clear()
            for i in range(len(unique_centers)):
                centers[i] = np.array(unique_centers[i])
            optimized = True
            for i in centers:
                optimized = True if np.array_equal(centers[i], previous_centers[i]) else False
                if not optimized:
                    # Break even if one center is not optimal
                    break
            if optimized:
                # Break out of main loop if all centers are optimal
                break
        # Make final set of centers accessible from instance
        self.centers = centers



def generate_data(iteration, axes = 2, bias = .04, bias_freq = 0):
    temp = []

    for i in range(iteration):
        if random.random() < bias_freq:
            factor = bias*100
        else:
            factor = 100
        datapoint = []
        for axis in range(axes):
            datapoint.append(int(random.random()*factor))
        temp.append(datapoint)
    data = np.array(temp)
    return data

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [8,2],
              [10,2],
              [9,3],])

clf = MeanShift()
clf.fit(X)

centers = clf.centers

plt.scatter(X[:, 0], X[:, 1], s=150)

for c in centers:
    plt.scatter(centers[c][0], centers[c][1], color='k', marker='*', s=150)

plt.show()


# iteration = int(input("Enter Data Amount: "))
#
# total_clusters = int(input("Enter Cluster Amount: "))
# data = generate_data(iteration)
# clf = K_Means(total_clusters)
# clf.fit(data)
#
# # Visualization & Testing
#
# for centroid in clf.centroids:
#     plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
#                 marker="o", color="k", s=50, linewidths=2)
# for group in clf.distinct_groups:
#     color = colors[group % 5]
#     for feature in clf.distinct_groups[group]:
#         plt.scatter(feature[0], feature[1], marker="x", color=color, s=50, linewidths=2)
#
# # Classify unknown data into existing clusters
# unknowns = generate_data(10)
# for unknown in unknowns:
#     classification = clf.predict(unknown)
#     color = colors[classification % 5]
#     plt.scatter(unknown[0], unknown[1], marker="*", color=color, s=25, linewidths=5)
#
# plt.show()





