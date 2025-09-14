import torch
import torch.nn.functional as F

class Kmeans:
    def __init__(self, n_clusters:int, init:str ='random', max_iters=10, p=1):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init = init
        self.p = p # 1 for L1 distance, 2 for ecudild distance

    def fit(self, X, cache_centroids=None):
        if cache_centroids is not None:
            centroids = cache_centroids
        else:
            centroids = self.init_centroids(X)
        for _ in range(self.max_iters):
            X_float = X.to(torch.float32)  # [1, 4096, 3072]
            centroids_float = centroids.to(torch.float32)  # [1, 32, 3072]
            dists = torch.cdist(X_float, centroids_float, p=self.p)  # [B, N, K]:[1, 4096, 32]
            labels = dists.argmin(dim=-1)  # [B, N]:[1, 4096]
            one_hot = F.one_hot(labels, num_classes=self.n_clusters).type_as(X)  # [B, N, K]:[1, 4096, 32]
            counts = one_hot.sum(dim=1) + 1e-8  # [B, K]:[1, 32]
            centroids_new = torch.bmm(
                one_hot.permute(0, 2, 1),  # [B, K, N]
                X                        # [B, N, D]
            ) / counts.unsqueeze(-1)         # [B, K, D]
            if torch.allclose(centroids, centroids_new, rtol=1e-4):
                break
            centroids = centroids_new
        return labels, centroids
    
    def init_centroids(self, X):
        B, N, dim = X.shape
        device = X.device
        if self.init == 'random':
            return X[torch.arange(B, device=device)[:, None], torch.randint(0, N, (B, self.n_clusters), device=device)]
        elif self.init == 'kmeans++':
            # didn't test yet
            return self.kmeans_plusplus(X)
        else:
            raise ValueError(f'Invalid init method: {self.init}')

    def kmeans_plusplus(self, X: torch.Tensor):
        '''
        fixed K-Means++
        '''
        B, N, dim = X.shape
        device = X.device
        centers = torch.zeros((B, self.n_clusters, dim), device=device)
        batch_indices = torch.arange(B, device=device)  # (B,)

        # sample the first center randomly
        sample_indices = torch.randint(0, N, (B,), device=device)  # (B,)
        centers[:, 0] = X[batch_indices, sample_indices]

        for i in range(1, self.n_clusters):
            dist = torch.cdist(X, centers[:, :i], p=1)  # (B, N, i)
            min_dist, _ = dist.min(dim=-1)  # (B, N)
            min_dist_sq = min_dist ** 2

            probs = min_dist_sq / (min_dist_sq.sum(dim=1, keepdim=True) + 1e-8)
            probs = probs / probs.sum(dim=1, keepdim=True)

            next_idx = torch.multinomial(probs, 1).squeeze(-1)  # (B,)
            centers[:, i] = X[batch_indices, next_idx]

        return centers
        