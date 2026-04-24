# Utility script for visualizing embeddings and evaluating clustering

import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, accuracy_score


def ensure_runs_dir():
    os.makedirs('runs', exist_ok=True)


def save_plot(xs, ys, labels, outpath, title):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(xs, ys, c=labels, cmap='tab20', s=10, alpha=0.8)
    plt.title(title)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main(quick=False, remove_top=0):
    ensure_runs_dir()
    t0 = time.strftime('%Y%m%d_%H%M%S')
    out_base = os.path.join('runs', f'embedding_vis_{t0}')
    os.makedirs(out_base, exist_ok=True)

    # Load embeddings and labels
    emb_path = os.path.join('embeddings', 'embeddings.npy')
    lbl_path = os.path.join('embeddings', 'labels.npy')
    meta_path = os.path.join('embeddings', 'metadata.json')
    if not os.path.exists(emb_path) or not os.path.exists(lbl_path):
        print('Could not find embeddings/labels under embeddings/. Run embed_dataset first.')
        return

    embeddings = np.load(emb_path)
    labels = np.load(lbl_path)

    # Quick sanity prints
    n_samples, dim = embeddings.shape
    n_classes = len(np.unique(labels))
    print(f'Loaded embeddings: {n_samples} samples, dim={dim}, classes={n_classes}')

    results = {}

    # PCA for explained variance and 2D plot
    pca50 = PCA(n_components=min(50, dim))
    proj50 = pca50.fit_transform(embeddings)
    evr = np.cumsum(pca50.explained_variance_ratio_)
    results['pca_explained_variance_cumsum'] = evr.tolist()

    # Optionally remove top-K PCA components (residualization) before downstream checks
    if remove_top and remove_top > 0:
        k = int(remove_top)
        if k >= proj50.shape[1]:
            print(f'remove_top={k} >= available PCA components ({proj50.shape[1]}), skipping removal')
        else:
            comps = pca50.components_[:k]  # k x D
            scores = proj50[:, :k]         # N x k
            # reconstruct top-k contribution and subtract
            topk_recon = scores.dot(comps)  # N x D
            embeddings = embeddings - topk_recon
            # recompute reduced PCA for plotting after removal
            pca2 = PCA(n_components=2)
            emb_pca2 = pca2.fit_transform(embeddings)
            save_plot(emb_pca2[:, 0], emb_pca2[:, 1], labels, os.path.join(out_base, f'pca2_removeTop{k}.png'), f'PCA (2D) removeTop{k}')
            print(f'Saved PCA plot with top-{k} components removed')
            # recompute small PCA for silhouette
            pca10 = PCA(n_components=min(10, dim))
            emb_pca10 = pca10.fit_transform(embeddings)
            try:
                sil = silhouette_score(emb_pca10, labels)
            except Exception:
                sil = None
            results[f'silhouette_pca10_removeTop{k}'] = float(sil) if sil is not None else None
    else:
        pca2 = PCA(n_components=2)
        emb_pca2 = pca2.fit_transform(embeddings)
        save_plot(emb_pca2[:, 0], emb_pca2[:, 1], labels, os.path.join(out_base, 'pca2.png'), 'PCA (2D)')
        print('Saved PCA plot')

    # If remove_top not applied earlier, compute silhouette on PCA10 now
    if not (remove_top and remove_top > 0):
        pca10 = PCA(n_components=min(10, dim))
        emb_pca10 = pca10.fit_transform(embeddings)
        try:
            sil = silhouette_score(emb_pca10, labels)
        except Exception:
            sil = None
        results['silhouette_pca10'] = float(sil) if sil is not None else None

    # Logistic regression baseline (train/test split)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42, stratify=labels)
    clf = LogisticRegression(max_iter=2000)
    try:
        clf.fit(X_train, y_train)
        ypred = clf.predict(X_test)
        acc = accuracy_score(y_test, ypred)
    except Exception as e:
        acc = None
        print('Logistic regression failed:', e)
    results['logistic_baseline_accuracy'] = float(acc) if acc is not None else None

    # TSNE and UMAP (if requested)
    if not quick:
        try:
            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', random_state=42)
            emb_tsne = tsne.fit_transform(embeddings)
            save_plot(emb_tsne[:, 0], emb_tsne[:, 1], labels, os.path.join(out_base, 'tsne.png'), 't-SNE (2D)')
            print('Saved t-SNE plot')
        except Exception as e:
            print('t-SNE failed or is slow:', e)

        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42)
            emb_umap = reducer.fit_transform(embeddings)
            save_plot(emb_umap[:, 0], emb_umap[:, 1], labels, os.path.join(out_base, 'umap.png'), 'UMAP (2D)')
            print('Saved UMAP plot')
        except Exception:
            print('UMAP not available; skipping.')

    # Write summary to runs/experiment_results.txt
    summary_path = os.path.join('runs', 'experiment_results.txt')
    # pick appropriate silhouette key (may be removeTop variant)
    sil_key = 'silhouette_pca10'
    if remove_top and remove_top > 0:
        candidate = f'silhouette_pca10_removeTop{int(remove_top)}'
        if candidate in results:
            sil_key = candidate

    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(f'=== Embedding visualization {t0} ===\n')
        f.write(f'n_samples={n_samples}, dim={dim}, n_classes={n_classes}\n')
        f.write(f'pca_explained_variance_cumsum (first 10): {evr[:10].tolist()}\n')
        f.write(f"{sil_key}: {results.get(sil_key)}\n")
        f.write(f"logistic_baseline_accuracy: {results['logistic_baseline_accuracy']}\n")
        f.write('\n')

    print('Summary written to', summary_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run quick mode (PCA + logistic baseline only)')
    parser.add_argument('--remove-top', type=int, default=0, help='Remove the top K PCA components from embeddings before analysis')
    args = parser.parse_args()
    main(quick=args.quick, remove_top=args.remove_top)
