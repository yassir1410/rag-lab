import math
from typing import List, Dict, Any, Optional

try:
    import numpy as np
except ImportError:
    np = None


class DimensionalityReducer:
    """
    Dimensionality reduction engine for dense vector spaces.
    Implements PCA, t-SNE, and UMAP for 2D and 3D projections with parameter validation.
    """

    @staticmethod
    def _fallback_pca(vectors: List[List[float]], n_components: int = 2) -> Dict[str, Any]:
        """
        Pure Python/NumPy Power-Iteration PCA fallback for small sets or when sklearn is unavailable.
        """
        n_samples = len(vectors)
        if n_samples == 0:
            return {"points": [], "explained_variance_ratio": [], "warnings": ["Empty input."]}

        dim = len(vectors[0])
        n_components = min(n_components, n_samples, dim)

        # Center data
        means = [sum(vectors[i][d] for i in range(n_samples)) / n_samples for d in range(dim)]
        centered = [[vectors[i][d] - means[d] for d in range(dim)] for i in range(n_samples)]

        if np is not None:
            arr = np.array(centered, dtype=float)
            # SVD decomposition
            u, s, vt = np.linalg.svd(arr, full_matrices=False)
            projected = np.dot(arr, vt[:n_components].T)
            total_var = float(np.sum(s**2))
            var_ratios = [
                float(s[i] ** 2 / total_var) if total_var > 0 else 0.0
                for i in range(n_components)
            ]
            points = [[round(float(coord), 4) for coord in pt] for pt in projected]
            for pt in points:
                while len(pt) < n_components:
                    pt.append(0.0)
            return {
                "points": points,
                "explained_variance_ratio": [round(r, 4) for r in var_ratios],
                "warnings": [],
            }

        # Fallback simple 2-axis projection using highest variance dimensions
        variances = [
            sum(centered[i][d] ** 2 for i in range(n_samples)) / max(n_samples, 1)
            for d in range(dim)
        ]
        top_dims = sorted(range(dim), key=lambda d: variances[d], reverse=True)[:n_components]
        points = [[round(centered[i][d], 4) for d in top_dims] for i in range(n_samples)]
        for pt in points:
            while len(pt) < n_components:
                pt.append(0.0)
        total_var = sum(variances) or 1.0
        var_ratios = [round(variances[d] / total_var, 4) for d in top_dims]
        return {
            "points": points,
            "explained_variance_ratio": var_ratios,
            "warnings": ["Using simplified variance projection fallback."],
        }

    def project(
        self,
        vectors: List[List[float]],
        algorithm: str = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Projects high-dimensional vectors to 2D or 3D.
        Returns projected points, metadata, and educational warnings.
        """
        n_samples = len(vectors)
        if n_samples == 0:
            return {
                "algorithm": algorithm,
                "n_components": n_components,
                "points": [],
                "warnings": ["No vectors provided for projection."],
            }

        dim = len(vectors[0])
        algo = algorithm.lower()
        warnings = []

        if n_samples < 4:
            warnings.append(
                f"Dataset size is very small ({n_samples} vectors). Dimensionality reduction "
                "requires sufficient sample density to preserve geometric topology. "
                "Distances and clusters will be largely arbitrary."
            )

        arr_vectors = np.array(vectors, dtype=float) if np is not None else vectors

        # 1. PCA
        if algo == "pca":
            if n_samples < 2:
                return {
                    "algorithm": "pca",
                    "n_components": n_components,
                    "points": [[0.0] * n_components],
                    "explained_variance_ratio": [1.0] + [0.0] * (n_components - 1),
                    "cumulative_variance": 1.0,
                    "warnings": warnings + ["Only 1 sample provided; cannot compute variance. Point placed at origin."],
                }
            try:
                from sklearn.decomposition import PCA
                k = min(n_components, n_samples, dim)
                pca = PCA(n_components=k, random_state=random_state)
                pts = pca.fit_transform(arr_vectors)
                points = [[round(float(c), 4) for c in row] for row in pts]
                for pt in points:
                    while len(pt) < n_components:
                        pt.append(0.0)
                var_ratios = [round(float(r), 4) for r in pca.explained_variance_ratio_]
                while len(var_ratios) < n_components:
                    var_ratios.append(0.0)
                return {
                    "algorithm": "pca",
                    "n_components": n_components,
                    "points": points,
                    "explained_variance_ratio": var_ratios,
                    "cumulative_variance": round(sum(var_ratios), 4),
                    "warnings": warnings,
                }
            except Exception:
                fallback_res = self._fallback_pca(vectors, n_components)
                fallback_res["warnings"].extend(warnings)
                fallback_res["algorithm"] = "pca"
                fallback_res["n_components"] = n_components
                return fallback_res

        # 2. t-SNE
        elif algo == "tsne":
            if n_samples < 2:
                return {
                    "algorithm": "tsne",
                    "n_components": n_components,
                    "points": [[0.0] * n_components],
                    "warnings": warnings + ["t-SNE requires at least 2 points."],
                }

            # Clamp perplexity: must be strictly less than n_samples
            adjusted_perplexity = min(perplexity, max(1.0, float(n_samples - 1)))
            if adjusted_perplexity != perplexity:
                warnings.append(
                    f"Adjusted t-SNE perplexity from {perplexity} to {adjusted_perplexity} "
                    f"because perplexity must be strictly less than sample count ({n_samples})."
                )

            try:
                from sklearn.manifold import TSNE
                k = min(n_components, n_samples)
                tsne = TSNE(
                    n_components=k,
                    perplexity=adjusted_perplexity,
                    random_state=random_state,
                    init="random",
                )
                pts = tsne.fit_transform(arr_vectors)
                points = [[round(float(c), 4) for c in row] for row in pts]
                for pt in points:
                    while len(pt) < n_components:
                        pt.append(0.0)
                warnings.append(
                    "t-SNE is non-convex and stochastic. Distances between clusters do not "
                    "reflect true global distances."
                )
                return {
                    "algorithm": "tsne",
                    "n_components": n_components,
                    "points": points,
                    "perplexity": adjusted_perplexity,
                    "warnings": warnings,
                }
            except Exception as e:
                warnings.append(f"t-SNE computation error ({e}), falling back to PCA.")
                pca_res = self.project(vectors, algorithm="pca", n_components=n_components)
                pca_res["warnings"].extend(warnings)
                return pca_res

        # 3. UMAP
        elif algo == "umap":
            if n_samples < 3:
                warnings.append("UMAP requires at least 3 points, falling back to PCA.")
                return self.project(vectors, algorithm="pca", n_components=n_components)

            adjusted_neighbors = min(n_neighbors, max(2, n_samples - 1))
            if adjusted_neighbors != n_neighbors:
                warnings.append(
                    f"Adjusted UMAP n_neighbors from {n_neighbors} to {adjusted_neighbors} "
                    f"due to small sample count ({n_samples})."
                )

            try:
                import umap
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=adjusted_neighbors,
                    min_dist=min_dist,
                    random_state=random_state,
                )
                pts = reducer.fit_transform(arr_vectors)
                points = [[round(float(c), 4) for c in row] for row in pts]
                for pt in points:
                    while len(pt) < n_components:
                        pt.append(0.0)
                return {
                    "algorithm": "umap",
                    "n_components": n_components,
                    "points": points,
                    "n_neighbors": adjusted_neighbors,
                    "min_dist": min_dist,
                    "warnings": warnings,
                }
            except Exception as e:
                warnings.append(f"UMAP computation unavailable or error ({e}), falling back to PCA.")
                pca_res = self.project(vectors, algorithm="pca", n_components=n_components)
                pca_res["warnings"].extend(warnings)
                return pca_res

        else:
            raise ValueError(f"Unknown dimensionality reduction algorithm: '{algorithm}'. Supported: 'pca', 'tsne', 'umap'.")


dimensionality_reducer = DimensionalityReducer()
