from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

x = np.random.random(13876)

km = KMeans()
y = km.fit(x.reshape(-1,1)) 

plt.figure(0)
plt.subplot(121); plt.plot(x)
plt.subplot(122); plt.plot(y)


plt.show()

