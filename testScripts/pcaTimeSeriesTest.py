import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pdb

#rng = np.random.RandomState(1)
t = np.arange(1000)/100.
x1 = np.sin(t) + t/10. + np.random.rand(len(t))/2.
x2 = np.sin(t) - t/10. + np.random.rand(len(t))/2.
x3 = t/10. + np.random.rand(len(t))/2.

# construct and plot original data
X = np.column_stack((x1,x2,x3))
meanX = np.mean(X,axis=0)
#X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.plot(t,x1)
plt.plot(t,x2)
plt.plot(t,x3)

# do PCA and plot principal components
pdb.set_trace()
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

plt.plot(t,X_pca[:,0],c='0.3')
plt.plot(t,X_pca[:,1],c='0.6')

print(pca.components_)
print(pca.explained_variance_)

# plot how original data is related principal components
comp = pca.components_
plt.plot(t,X_pca[:,0]*comp[0,0]+X_pca[:,1]*comp[1,0]+meanX[0])


# reconstract original data without 1st principal component
Xhat = np.dot(pca.transform(X)[:,1:], pca.components_[1:,:])
Xhat += meanX

plt.plot(t,Xhat[:,0],ls='--')

plt.show()
pdb.set_trace()

# PCA as dimensionality reduction
# project data to
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)
X_new = pca.inverse_transform(X_pca)



plt.plot(t,X_new)
plt.show()
