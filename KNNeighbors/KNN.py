import random
import math
import heapq

# generate random number by normal Gaussian distribution
def gaussCluster(center, stdDev, count=50):
        a=[(random.gauss(center[0], stdDev),random.gauss(center[1], stdDev)) for _ in range(count)]
        return a;

# test: generate and store tow groups of random number by <-4,0> of mean, 1 of variance and <4,0> of mean and 1 of variance
def makeDummyData():
    return gaussCluster((-4,0), 1) + gaussCluster((4,0), 1)

# metric of distance is using EuclideanDistance
def euclideanDistance(x,y):
    return math.sqrt(sum([(a-b)**2 for (a,b) in zip(x,y)]))

# core of KNN
def makeKNNClassifier(data, labels, k, distance):
    def classify(x):
        closestPoints = heapq.nsmallest(k, enumerate(data),key=lambda y: distance(x, y[1]))
        closestLabels = [labels[i] for (i,temp) in closestPoints]
#         print 'set():', set(closestLabels)
#         print 'key'
        return max(set(closestLabels), key=closestLabels.count)
 
    return classify

# make data and labels
trainingPoints = makeDummyData() # has 50 points from each class
trainingLabels = [1] * 50 + [2] * 50  # an arbitrary choice of labeling
 
f = makeKNNClassifier(trainingPoints, trainingLabels, 8, euclideanDistance)
print f((0,0))
