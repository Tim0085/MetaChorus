from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any, Mapping

from m9_common import load_json


def tsne_kwargs(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    from sklearn.manifold import TSNE

    kwargs = {
        "n_components": 2,
        "perplexity": float(config["perplexity"]),
        "early_exaggeration": float(config["early_exaggeration"]),
        "learning_rate": float(config["learning_rate"]),
        "init": config["initialization"],
        "metric": config["metric"],
        "random_state": int(seed),
    }
    parameter = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    kwargs[parameter] = int(config["iterations"])
    return kwargs


def reduce_embedding(embedding: Any, config: Mapping[str, Any], seed: int) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE, trustworthiness

    if int(config["perplexity"]) >= len(embedding):
        raise ValueError("t-SNE perplexity must be lower than n_samples")
    dimensions = min(int(config["pca_dimensions"]), int(embedding.shape[1]))
    pca = PCA(n_components=dimensions, svd_solver="full")
    projected = pca.fit_transform(embedding)
    kwargs = tsne_kwargs(config, seed)
    model = TSNE(**kwargs)
    raw = model.fit_transform(projected)
    centered = raw - raw.mean(axis=0, keepdims=True)
    scale = float(np.max(np.abs(centered)))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("invalid t-SNE display scale")
    display = centered / scale
    diagnostics = {
        "visualization_seed": int(seed),
        "kl_divergence": float(model.kl_divergence_),
        "n_iter": int(getattr(model, "n_iter_", config["iterations"])),
        "trustworthiness_at_5": float(trustworthiness(projected, raw, n_neighbors=5)),
        "trustworthiness_at_15": float(trustworthiness(projected, raw, n_neighbors=15)),
        "trustworthiness_at_30": float(trustworthiness(projected, raw, n_neighbors=30)),
        "pca_dimensions": dimensions,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "tsne_parameter_name_for_iterations": "max_iter" if "max_iter" in kwargs else "n_iter",
        "tsne_parameters": kwargs,
        "display_transform": "center_then_isotropic_max_abs_scale",
    }
    return (raw, display), diagnostics


def highdim_diagnostics(embedding: Any, labels: Any, *, kmeans_seed: int = 2025) -> dict[str, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    assignment = KMeans(
        n_clusters=len(set(labels.tolist())), random_state=kmeans_seed,
        n_init=10, algorithm="lloyd",
    ).fit_predict(embedding)
    return {
        "silhouette_label_highdim_euclidean": float(silhouette_score(embedding, labels, metric="euclidean")),
        "silhouette_label_highdim_cosine": float(silhouette_score(embedding, labels, metric="cosine")),
        "silhouette_kmeans_highdim_euclidean": float(silhouette_score(embedding, assignment, metric="euclidean")),
        "kmeans_diagnostic_seed": int(kmeans_seed),
    }


def main() -> None:
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    embedding = np.load(args.embedding, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    config = load_json(args.config)
    (_, _), tsne = reduce_embedding(embedding, config, args.seed)
    print({**highdim_diagnostics(embedding, labels), **tsne})


if __name__ == "__main__":
    main()
