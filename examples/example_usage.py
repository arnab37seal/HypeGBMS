import numpy as np
from hyperbolic_gbms.core import HyperbolicGBMS

# simple example run
if __name__ == "__main__":
    X = np.random.randn(20, 3)
    model = HyperbolicGBMS(sigma=0.6, max_iter=5)
    model.fit(X)
    print("Number of clusters:", model.n_clusters_)
