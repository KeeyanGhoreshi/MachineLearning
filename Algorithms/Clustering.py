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
    def __init__(self, radius=None, step = 8):
        self.radius = radius
        self.step = step
    def fit(self, data):
        if self.radius == None:
            main_center = np.average(data, axis=0)
            main_center_distance = np.linalg.norm(main_center)
            self.radius = main_center_distance/self.step
            print(self.radius)
        centers = {}
        for i in range(len(data)):
            # Make every data point a center
            centers[i] = data[i]
        weights = [i for i in range(self.step)][::-1]
        while True:
            new_centers = []
            for i in centers:
                center = centers[i]
                within_radius = []
                numerator = 0
                denom = 0
                for point in data:
                    # We can use a gaussian kernel for distance weight
                    distance = np.linalg.norm(point-center)
                    rbf_k=np.exp(-(distance ** 2)/self.step)
                    # Find weighted average
                    numerator+= rbf_k*point
                    denom+=rbf_k

                # Make new center the mean of features within the radius (bandwidth)
                new_centers.append(tuple(numerator/denom))
            unique_centers = sorted(list(set(new_centers)))     # Combine centers with the same location
            to_remove = []
            for i in unique_centers:
                for ii in [i for i in unique_centers]:
                    if i==ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii))<=self.radius:
                        to_remove.append(ii)
                        break
            for i in to_remove:
                try:
                    unique_centers.remove(i)
                except:
                    pass
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
        self.classes = {}
        for i in range(len(self.centers)):
            self.classes[i] = []
        for point in data:
            distances = [np.linalg.norm(point - self.centers[center]) for center in self.centers]
            classification = (distances.index(min(distances)))
            self.classes[classification].append(point)


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
plt.figure(1)
centers = clf.centers
for classification in clf.classes:
    color = colors[classification % 6]
    for featureset in clf.classes[classification]:
        plt.scatter(featureset[0],featureset[1], marker = "x", color=color, s=150, linewidths = 5, zorder = 10)
for c in centers:
    plt.scatter(centers[c][0], centers[c][1], color='k', marker='*', s=150)



iteration = int(input("Enter Data Amount: "))

total_clusters = int(input("Enter Cluster Amount: "))
data = generate_data(iteration)
clf = K_Means(total_clusters)
clf.fit(data)

# Visualization & Testing
plt.figure(2)
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





