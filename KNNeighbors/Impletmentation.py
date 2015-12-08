import random
import math
import heapq

def gaussCluster(center, stdDev, count=50):
        a=[(random.gauss(center[0], stdDev),random.gauss(center[1], stdDev)) for _ in range(count)]
        return a;
    
def makeDummyData():
    return gaussCluster((-4,0), 1) + gaussCluster((4,0), 1)

def euclideanDistance(x,y):
    #print zip(x,y)
    return math.sqrt(sum([(a-b)**2 for (a,b) in zip(x,y)]))

def makeKNNClassifier(data, labels, k, distance):
    def classify(x):
        closestPoints = heapq.nsmallest(k, enumerate(data),key=lambda y: distance(x, y[1]))
        closestLabels = [labels[i] for (i, pt) in closestPoints]
        return max(set(closestLabels), key=closestLabels.count)
 
    return classify

trainingPoints = makeDummyData() # has 50 points from each class
trainingLabels = [1] * 50 + [2] * 50  # an arbitrary choice of labeling
 
f = makeKNNClassifier(trainingPoints, trainingLabels, 8, euclideanDistance)
print f((1,1))