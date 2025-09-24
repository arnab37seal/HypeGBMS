import torch
import time
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def safe_norm(x, dim=-1, keepdim=True, eps=1e-10):
    return torch.norm(x, dim=dim, keepdim=keepdim).clamp(min=eps)

def project_to_poincare_ball(X, c=-1.0):
    norm = torch.norm(X, dim=-1, keepdim=True)
    sqrt_c = torch.sqrt(torch.tensor(-c, dtype=X.dtype, device=X.device))
    scaling = torch.tanh(sqrt_c * norm) / (sqrt_c * norm + 1e-10)
    return X * scaling

def poincare_distance(a, b, c=-1.0):
    norm_a = torch.norm(a, dim=-1)
    norm_b = torch.norm(b, dim=-1)
    norm_diff = torch.norm(a - b, dim=-1)
    denom = (1 - c * norm_a**2) * (1 - c * norm_b**2)
    denom = torch.clamp(denom, min=1e-10)
    inner = 1 + (2 * norm_diff**2) / denom
    inner = torch.clamp(inner, min=1 + 1e-10)
    return torch.acosh(inner) / torch.sqrt(torch.tensor(-c, dtype=a.dtype, device=a.device))

def mobius_add(a, b, c=-1.0):
    ab = torch.sum(a * b, dim=-1, keepdim=True)
    norm_a = torch.sum(a * a, dim=-1, keepdim=True)
    norm_b = torch.sum(b * b, dim=-1, keepdim=True)
    denom = 1 - 2 * c * ab + c**2 * norm_a * norm_b
    denom = denom.clamp(min=1e-10)
    result = ((1 - 2 * c * ab - c * norm_b) * a + (1 + c * norm_a) * b) / denom
    norm_result = torch.norm(result, dim=-1, keepdim=True)
    result = torch.where(norm_result >= 1.0, result * (0.999 / norm_result), result)
    return result

def mobius_scalar_mult(r, v, c=-1.0):
    norm_v = torch.norm(v, dim=-1, keepdim=True)
    sqrt_c = torch.sqrt(torch.tensor(-c, dtype=v.dtype, device=v.device))
    x = torch.clamp(sqrt_c * norm_v, min=-1 + 1e-6, max=1 - 1e-6)
    tanh_part = torch.tanh(r * torch.atanh(x))
    return (tanh_part / (sqrt_c * norm_v + 1e-10)) * v

def log_map(x, y, c=-1.0):
    diff = mobius_add(-x, y, c=c)
    norm_diff = torch.norm(diff, dim=-1, keepdim=True)
    sqrt_c = torch.sqrt(torch.tensor(-c, dtype=x.dtype, device=x.device))
    norm_x = torch.norm(x, dim=-1, keepdim=True)
    lambda_x = 2 / (1 - c * norm_x**2).clamp(min=1e-10)
    scale = (2 / (sqrt_c * lambda_x)) * torch.atanh(sqrt_c * norm_diff.clamp(max=1 - 1e-5))
    return (scale / (norm_diff + 1e-10)) * diff

def exp_map(x, v, c=-1.0):
    norm_v = torch.norm(v, dim=-1, keepdim=True)
    sqrt_c = torch.sqrt(torch.tensor(-c, dtype=v.dtype, device=v.device))
    norm_x = torch.norm(x, dim=-1, keepdim=True)
    lambda_x = 2 / (1 - c * norm_x**2).clamp(min=1e-10)
    scale = torch.tanh(sqrt_c * lambda_x * norm_v / 2) / (sqrt_c * norm_v + 1e-10)
    v_scaled = scale * v
    return mobius_add(x, v_scaled, c=c)

def frechet_mean(points, c=-1.0, max_iter=20, tol=1e-5):
    mu = points[0].clone()
    for _ in range(max_iter):
        logs = torch.stack([log_map(mu, p.unsqueeze(0), c=c).squeeze() for p in points])
        mean_tangent = torch.mean(logs, dim=0, keepdim=True)
        move = torch.norm(mean_tangent)
        mu = exp_map(mu, mean_tangent, c=c).squeeze()
        if move < tol:
            break
    return mu

def mobius_weighted_mean_batch(X, W, c=-1.0):

    N, D = X.shape
    sqrt_c = torch.sqrt(torch.tensor(-c, dtype=X.dtype, device=X.device))

    W_sum = torch.sum(W, dim=1, keepdim=True) + 1e-9
    W_norm = W / W_sum  
    X_exp = X.unsqueeze(0).expand(N, -1, -1) 
    W_exp = W_norm.unsqueeze(2)              
    X_scaled = mobius_scalar_mult(W_exp, X_exp, c=c)  

    mean = X_scaled[:, 0, :]
    for j in range(1, N):
        mean = mobius_add(mean, X_scaled[:, j, :], c=c)  

    return mean

class HyperbolicGBMS:
    def __init__(self, sigma=0.4, tol=1e-5, min_diff=1e-4, max_iter=300, curvature=-1.0, verbose=True, random_state=None):
        self.sigma = sigma
        self.tol = tol
        self.min_diff = min_diff
        self.max_iter = max_iter
        self.c = curvature
        self.verbose = verbose
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_input):
        X = torch.tensor(X_input, dtype=torch.float32, device=self.device)
        X = project_to_poincare_ball(X, c=self.c)
        N, D = X.shape

        self.history_ = [X.clone()]
        self.entropy_ = []
        self.dist_times_ = []
        old_entropy = -1000

        for it in range(self.max_iter):
            X_prev = X.clone()

            dist_start = time.time()
            dists_squared = poincare_distance(X.unsqueeze(1), X.unsqueeze(0), c=self.c).pow(2)
            dist_duration = time.time() - dist_start
            self.dist_times_.append(dist_duration)

            W = torch.exp(-dists_squared / (2 * self.sigma ** 2))

           
            X = mobius_weighted_mean_batch(X, W, c=self.c)
            X = torch.clamp(X, min=-1 + 1e-5, max=1 - 1e-5)
            self.history_.append(X.clone())

            point_dists = poincare_distance(X_prev, X, c=self.c)
           
            displacements = log_map(X_prev, X, c=self.c)
            avg_movement = torch.mean(torch.norm(displacements, dim=-1)).item()

            dists = point_dists.detach().cpu().numpy()
            B = int(0.9 * len(dists))
            hist, _ = np.histogram(dists, bins=B, range=(0, np.max(dists) + 1e-9))
            hist = hist / (np.sum(hist) + 1e-9)
            entropy = -np.sum(hist[hist > 0] * np.log(hist[hist > 0]))

            self.entropy_.append(entropy)
            entropy_change = abs(entropy - old_entropy)
            old_entropy = entropy

            if self.verbose:
                print(f"Iter {it+1}: avg_move = {avg_movement:.4e}, entropy = {entropy:.4f}, ∆H = {entropy_change:.2e}, distance_time = {dist_duration:.4f}s")

            if avg_movement < self.tol or entropy_change < 1e-8:
                break

        self.X_final_ = X.detach().cpu().numpy()
        self._graph_cluster(X)
        return self

    def _graph_cluster(self, X):
        X_np = X.detach().cpu().numpy()
        final_dists = pairwise_distances(X_np)
        adj = (final_dists < self.min_diff).astype(int)
        np.fill_diagonal(adj, 1)
        graph = csr_matrix(adj)
        _, component_ids = connected_components(csgraph=graph, directed=False)

        self.labels_ = component_ids
        self.n_clusters_ = len(set(component_ids))
        self.clust_ = {
            i: np.where(component_ids == i)[0].tolist()
            for i in range(self.n_clusters_)
        }
        self.centroids_ = torch.stack([
            frechet_mean(X[component_ids == i], c=self.c)
            for i in range(self.n_clusters_)
        ]) if self.n_clusters_ > 0 else torch.empty(0, X.shape[1])

