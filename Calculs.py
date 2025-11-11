from math import *
from matplotlib import pylab as plt
import numpy as np

import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

class Calculs(object):
    def Minkowski(self, A, B, p = 3):
        res = 0
        for i in range(len(A)):
            res += (abs(A[i] - B[i]) ** p)
        return res ** (1 / p)

    def Manhattan(self, A, B, col = {}):
        res = 0
        for i in range(len(A)):
            calc = abs(A[i] - B[i])
            col[i + 1] = calc
            res += calc
        return res
    
    def Euclidienne(self, A, B, col = {}):
        res = 0
        for i in range(len(A)):
            calc = (A[i] - B[i]) ** 2
            col[i + 1] = calc
            res += calc
        return sqrt(res)
          
    def Chebyshev(self, A, B, col = {}):
        max = 0
        for i in range(len(A)):
            calc = abs(A[i] - B[i])
            col[i + 1] = calc
            if calc > max:
                max = calc 
        return max
    
    def regression(self, col, k):
        res = 0
        for val in col:
            res += col[val]
        return res / k

    def classification(self, col, k):
        occurrences = {}

        for i in range(1, k + 1):
            if col[i - 1] in occurrences:
                occurrences[col[i - 1]] += 1
            else:
                occurrences[col[i - 1]] = 1

        leMaxi = None
        maxi = 0
        for key, value in occurrences.items():
            if value > maxi:
                maxi = value
                leMaxi = key

        return leMaxi

    def toGraphe(self, col = {}):
        plt.figure('Distances entre A et B')
        col = np.array(list(col.items()))
        plt.clf()
        plt.plot(col, color='blue', marker='o', linestyle='None')
        plt.ylabel('Distance')
        plt.xlabel('Classement')
        plt.savefig('mafig.pdf')

    def tri(self, col = {}):
        temp = sorted(col.values())
        for i in range(1, len(col) + 1):
            del col[i]
        for j in range(1, len(temp) + 1):
            col[j] = temp[j - 1]
        return col

    def normaliser(self, val):
        scaler = MinMaxScaler()
        res = scaler.fit_transform(val)
        return res
    
    def listeToMatr(self, x, y):
        matr = []
        for i in range(len(x)):
            matr.append([x[i], y[i]])
        return matr                 

    def kmeans(matr, k, max_iters=100, tol=1e-4):
        indices = np.random.choice(len(matr), k, replace=False)
        centroids = matr[indices]

        for _ in range(max_iters):
            distances = np.linalg.norm(matr[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([matr[labels == i].mean(axis=0) for i in range(k)])
            
            if np.linalg.norm(new_centroids - centroids) < tol:
                break
            centroids = new_centroids
            
        return labels, centroids
