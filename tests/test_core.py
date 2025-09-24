import numpy as np
from hyperbolic_gbms.core import HyperbolicGBMS

X = np.random.randn(100, 3)
model = HyperbolicGBMS(sigma=0.6, max_iter=50)
model.fit(X)
print("Cluster labels:", model.labels_)
