n = int(input())
X = []
for i in range(n):
    X.append([float(x) for x in input().split()])

import numpy as np
X = np.array(X).reshape(-1,2) # 2D array

# define initial centroids as 1D arrays
C1_init = np.array([0, 0])
C2_init = np.array([2, 2])

C1_set = []
C2_set = []
# assign each imput coordinate to either first or second initial centroid
for i in range(n):
    # calculate Euclidean distances
    dist1 = np.sqrt(((X[i]-C1_init)**2).sum())
    dist2 = np.sqrt(((X[i]-C2_init)**2).sum())
    if dist1 <= dist2: # compare distances
        C1_set.append(X[i])
    else:
        C2_set.append(X[i])

C1_set = np.array(C1_set).reshape(-1,2) # 2D arrays again
C2_set = np.array(C2_set).reshape(-1,2)

# calculate and print new first centroid
if C1_set.size > 0:  # ?could use len(C1_set) instead? but this definitely works
    print(np.mean(C1_set, axis=0).round(2)) # in proscribed format
else:
    print(None) # to pass Test 3

# repeat for second initial centroid
if C2_set.size > 0:
    print(np.mean(C2_set, axis=0).round(2))
else:
    print(None)