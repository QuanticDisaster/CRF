
import pandas as pd
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

df = pd.read_csv(r'../data/segment.txt', sep=" ")
data = df.to_numpy()[:10000000]
labels_score = data[:,-5:] # float array : [N_points, N_classes]

print("adding unary terms")
d = dcrf.DenseCRF(len(labels_score), labels_score.shape[1])  # npoints, nlabels
d.setUnaryEnergy(unary_from_softmax(np.exp(labels_score).T).astype(np.float32))

print("adding pairwise terms")
#use the positions as features (smoothness kernel in the readme of pydenseCRF)
feats = data[:,0:3].T * 5 # standard deviation is directly applied to circumvent the absence of argument in PairWiseEnergy function
feats = feats.astype(np.float32)
compatibility = np.array([[-1,      0,  0,  0,  0], # when considering road marking point, the high presence of road points around is not considered as a big mistake (-0.9)
                          [-0.9,    -1, 0,  0,  0],
                          [0,       0, -1,  0,  0],
                          [-0.5,    0,  0,  -1, 0],
                          [-0.5,    0,  0,  0,  -1]]).astype(np.float32)
weight = 10
d.addPairwiseEnergy(feats, compat=weight * compatibility, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

# add intensity (don't forget we need to re add positions to have a bilateral kernel)
#feats = np.hstack([data[:,0:3], data[:,-7].reshape(-1,1)]).T.astype(np.float32)  # Get the pairwise features from somewhere, shape (feature dimensionality, npoints), dtype('float32')
#d.addPairwiseEnergy(feats, compat=0.5, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NO_NORMALIZATION )


print("starting inference")
Q = d.inference(5)
proba = np.array(Q)

# print some informations
new_label = proba.T
diff = np.argmax(labels_score, axis=1) != np.argmax(new_label, axis=1)
print("number of points changing class : {}/{}".format(diff.sum(), len(labels_score)))

change_matrix = np.zeros((labels_score.shape[1], labels_score.shape[1]))
for i in range(labels_score.shape[1]):
    for j in range(labels_score.shape[1]):
        if i == j:
            continue
        change_matrix[i,j] = (np.logical_and(np.argmax(new_label, axis=1) == j, np.argmax(labels_score, axis=1) == i)).sum()
np.set_printoptions(suppress=True) # disable scientific notation for printing of np arrays     
print("number of points converted from one class (line) to another class (column) \n", change_matrix)

# save result to file with an additionnal column for updated labels
data = np.hstack([data, np.argmax(new_label, axis=1).reshape(-1,1)])
np.savetxt(r'segment_smoothed.txt', data, fmt='%1.3f')


# questions :
# be careful with compat and the sign. Using compat=X is equivalent to using compat = - X * np.ones. Maybe open an issue about it
# it seems the standard deviation (theta on the readme) can't be given wwhen used with pairwiseEnergy. Maybe we can circumvent it by divinding directly the features.
# using compat = np.array([0,1,1,1,1]).astype(np.float32) for 5 labels for instance makes that all points belonging to the first class won't be updated at all