import numpy as np
from hyperbolic_gbms.core import HyperbolicGBMS

def test_fit_runs():
    X = np.random.randn(10, 3)
    model = HyperbolicGBMS(max_iter=2, verbose=False)
    model.fit(X)
    assert hasattr(model, "labels_")
    assert len(model.labels_) == X.shape[0]
