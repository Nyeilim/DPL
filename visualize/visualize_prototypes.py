#!/usr/bin/env python3
"""
Script to visualize DPL prototypes and samples using PCA visualization.

This script loads previously extracted prototype embeddings and sample embeddings
from pkl file, and creates PCA visualizations with sigma uncertainty circles.
No need to load the model - everything is already saved in the pkl file.
"""

import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
from pysgg.data import get_dataset_statistics
from pysgg.config import cfg

# Configuration
embeddings_path = "/root/autodl-tmp/visualize/relation_embeddings.pkl"
config_path = "/root/autodl-tmp/ckpt/config.yml"
output_dir = "/root/autodl-tmp/visualize"
do_3d = True

# ========== 可在此处修改要显示的谓词列表 ==========
# 选中要可视化的谓词（修改下面列表）
SELECTED_PREDICATES = [
    'on', 'holding', 'of', 'across', 'wearing',
    # 'has', 'made of',
    # 'using', 'lying on', 'walking on'
]
# ==================================================

# 每个谓词选取的最靠近原型的样本数量
MAX_SAMPLES_PER_PREDICATE = 50


def load_embeddings_data(embeddings_path):
    """Load embeddings and prototypes from file."""
    print(f"Loading embeddings data from: {embeddings_path}")

    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)

    # Check if data has the new format with prototypes
    if isinstance(data, dict) and 'sample_embeddings' in data:
        sample_embeddings = data['sample_embeddings']
        prototypes_data = data.get('prototypes')
        print(f"Loaded {len(sample_embeddings)} sample embeddings")
        if prototypes_data is not None:
            print(f"Loaded prototypes: {prototypes_data['proto_emb'].shape}")
        else:
            print("No prototypes found in file")
    else:
        # Old format - just sample embeddings
        sample_embeddings = data
        prototypes_data = None
        print(f"Loaded {len(sample_embeddings)} sample embeddings (old format)")

    return sample_embeddings, prototypes_data


def load_predicate_labels(config_path):
    """Get predicate class labels from config."""
    print(f"Loading config from: {config_path}")
    cfg.merge_from_file(config_path)

    statistics = get_dataset_statistics(cfg)
    rel_classes = statistics["rel_classes"]

    # Remove background class if present
    if rel_classes[0] == "__background__":
        rel_classes = rel_classes[1:]

    print(f"Loaded {len(rel_classes)} predicate labels")
    return rel_classes


def filter_samples(sample_embeddings, predicate_labels_all, proto_emb, predicate_to_idx):
    """Filter samples to only selected predicates, selecting top MAX_SAMPLES_PER_PREDICATE samples closest to their prototype."""
    print(f"\nFiltering samples...")
    print(f"Total predicates available: {len(predicate_labels_all)}")
    print(f"Selected predicates: {SELECTED_PREDICATES}")
    print(f"Max samples per predicate: {MAX_SAMPLES_PER_PREDICATE}")

    # Group samples by predicate
    predicate_samples = {pred: [] for pred in SELECTED_PREDICATES}

    for sample in sample_embeddings:
        gt_label_idx = sample['gt_label']
        if gt_label_idx >= len(predicate_labels_all):
            continue
        predicate = predicate_labels_all[gt_label_idx]

        # Only keep if predicate is in selected list
        if predicate in SELECTED_PREDICATES:
            predicate_samples[predicate].append(sample)

    # For each predicate, select top samples closest to prototype
    filtered_samples = []
    counts = {pred: 0 for pred in SELECTED_PREDICATES}

    for predicate in SELECTED_PREDICATES:
        samples = predicate_samples[predicate]
        if not samples:
            continue

        # Get prototype index
        proto_idx = predicate_to_idx.get(predicate)
        if proto_idx is None:
            continue

        prototype = proto_emb[proto_idx]

        # Calculate distances to prototype
        sample_vecs = np.array([s['embedding'] for s in samples])
        sample_norm = sample_vecs / np.linalg.norm(sample_vecs, axis=1, keepdims=True)
        proto_norm = prototype / np.linalg.norm(prototype)

        # Compute cosine similarity (dot product for normalized vectors)
        similarities = np.dot(sample_norm, proto_norm)

        # Sort by distance (1 - similarity), pick closest (highest similarity)
        sorted_indices = np.argsort(-similarities)  # Descending similarity
        top_indices = sorted_indices[:min(len(samples), MAX_SAMPLES_PER_PREDICATE)]

        # Add top samples to filtered list
        for idx in top_indices:
            filtered_samples.append(samples[idx])
            counts[predicate] += 1

    # Print statistics
    print(f"\nFiltered sample counts:")
    for pred in SELECTED_PREDICATES:
        print(f"  {pred}: {counts[pred]} samples")

    print(f"\nTotal filtered samples: {len(filtered_samples)}")

    # Debug: print samples that were filtered out
    print(f"\nSamples filtered out (not in SELECTED_PREDICATES):")
    filtered_out_counts = {}
    for sample in sample_embeddings:
        gt_label_idx = sample['gt_label']
        if gt_label_idx >= len(predicate_labels_all):
            continue
        predicate = predicate_labels_all[gt_label_idx]
        if predicate not in SELECTED_PREDICATES:
            filtered_out_counts[predicate] = filtered_out_counts.get(predicate, 0) + 1
    if filtered_out_counts:
        for pred, count in sorted(filtered_out_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {pred}: {count} samples")

    return filtered_samples


def visualize_prototypes_pca(proto_emb, sigma, sample_embeddings, predicate_labels_filtered, predicate_labels_all, output_dir, n_components=2):
    """Visualize prototypes and samples using PCA with sigma uncertainty circles."""

    # Normalize prototypes
    proto_norm = proto_emb / np.linalg.norm(proto_emb, axis=1, keepdims=True)

    # Collect sample embeddings
    sample_vecs = np.array([item['embedding'] for item in sample_embeddings])
    sample_labels_idx = [item['gt_label'] for item in sample_embeddings]

    # Get predicate names from original labels
    sample_predicates = [predicate_labels_all[idx] if idx < len(predicate_labels_all) else "unknown"
                       for idx in sample_labels_idx]

    # Normalize sample embeddings
    sample_norm = sample_vecs / np.linalg.norm(sample_vecs, axis=1, keepdims=True)

    # Apply PCA
    print(f"\nApplying PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    proto_pca = pca.fit_transform(proto_norm)
    sample_pca = pca.transform(sample_norm)

    # Transform sigma to PCA space
    # For each dimension, compute the variance in PCA space
    # We'll use average sigma as radius
    avg_sigma = np.mean(sigma, axis=1)  # (num_predicates,)
    sigma_pca = avg_sigma  # Use average sigma as radius

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Create visualization
    plt.figure(figsize=(16, 12))

    # Use different colors for different predicates - use high contrast palette
    n_predicates = len(predicate_labels_filtered)
    # Use Set1 for higher contrast colors, cycle if needed
    cmap = plt.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, max(n_predicates, len(SELECTED_PREDICATES))))

    # Plot samples first (larger, more visible points)
    for i, (label_idx, point) in enumerate(zip(sample_predicates, sample_pca)):
        # Find which predicate this sample belongs to in the filtered list
        predicate = sample_predicates[i]
        if predicate in predicate_labels_filtered:
            color_idx = predicate_labels_filtered.index(predicate)
            plt.scatter(point[0], point[1], c=colors[color_idx], s=50,
                      edgecolors='black', linewidth=0.3, alpha=1.0,
                      marker='x', zorder=5)  # Draw under prototypes

    # Plot each predicate prototype (larger points)
    for i, (label, point) in enumerate(zip(predicate_labels_filtered, proto_pca)):
        # Plot prototype point
        plt.scatter(point[0], point[1], c=colors[i], s=250,
                  edgecolors='black', linewidth=1.5, alpha=0.9,
                  marker='o', zorder=10)  # Draw on top

    # Create color-predicate mapping legend (display actual colors)
    legend_elements = []
    for i, label in enumerate(predicate_labels_filtered):
        legend_elements.append(
            Line2D([0], [0], marker='s', color=colors[i], markersize=8,
                   linestyle='None', label=label)
        )

    # Add prototype/sample markers to legend
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='black', markersize=10,
                linestyle='None', label='Prototype'),
        Line2D([0], [0], marker='x', color='black', markersize=6,
                linestyle='None', label='Sample')
    ])

    # Place legend inside the plot (right side, outside to avoid overlap)
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
               fontsize=9, framealpha=0.9)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    plt.title('DPL Prototypes and Samples - PCA Visualization',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(output_dir, 'prototypes_pca.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"PCA visualization saved to: {save_path}")

    # Save PCA data
    data_save_path = os.path.join(output_dir, 'prototypes_pca_data.pkl')
    pca_data = {
        'proto_pca': proto_pca,
        'sample_pca': sample_pca,
        'sigma_pca': sigma_pca,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'predicate_labels': predicate_labels_filtered,
        'proto_emb': proto_emb,
        'sigma': sigma
    }
    with open(data_save_path, 'wb') as f:
        pickle.dump(pca_data, f)
    print(f"PCA data saved to: {data_save_path}")

    plt.show()


def visualize_prototypes_3d_pca(proto_emb, sigma, sample_embeddings, predicate_labels_filtered, predicate_labels_all, output_dir):
    """Create 3D PCA visualization with sigma uncertainty spheres."""

    # Normalize prototypes
    proto_norm = proto_emb / np.linalg.norm(proto_emb, axis=1, keepdims=True)

    # Collect and normalize sample embeddings
    sample_vecs = np.array([item['embedding'] for item in sample_embeddings])
    sample_labels_idx = [item['gt_label'] for item in sample_embeddings]

    # Get predicate names from original labels
    sample_predicates = [predicate_labels_all[idx] if idx < len(predicate_labels_all) else "unknown"
                       for idx in sample_labels_idx]

    # Normalize sample embeddings
    sample_norm = sample_vecs / np.linalg.norm(sample_vecs, axis=1, keepdims=True)

    # Apply 3D PCA
    print(f"\nApplying 3D PCA...")
    pca = PCA(n_components=3)
    proto_pca = pca.fit_transform(proto_norm)
    sample_pca = pca.transform(sample_norm)

    # Get average sigma
    avg_sigma = np.mean(sigma, axis=1)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Create 3D visualization
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')

    n_predicates = len(predicate_labels_filtered)
    # Use Set1 for higher contrast colors
    cmap = plt.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, max(n_predicates, len(SELECTED_PREDICATES))))

    # Plot samples first (smaller points)
    for i, point in enumerate(sample_pca):
        predicate = sample_predicates[i]
        if predicate in predicate_labels_filtered:
            color_idx = predicate_labels_filtered.index(predicate)
            ax.scatter(point[0], point[1], point[2], c=[colors[color_idx]], s=20,
                     marker='x', alpha=0.6, zorder=5)

    # Plot each predicate prototype (larger points)
    for i, (label, point) in enumerate(zip(predicate_labels_filtered, proto_pca)):
        # Plot prototype point
        ax.scatter(point[0], point[1], point[2], c=[colors[i]], s=150,
                 edgecolors='black', linewidth=1, alpha=0.9, zorder=10)

        # No label text to avoid clutter

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=12)
    ax.set_title('DPL Prototype Embeddings - 3D PCA Visualization',
                 fontsize=14, fontweight='bold')

    # Create legend for predicates
    legend_elements = []
    for i, label in enumerate(predicate_labels_filtered):
        legend_elements.append(
            Line2D([0], [0], marker='s', color=colors[i], markersize=8,
                   linestyle='None', label=label)
        )

    # Add marker legend
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='black', markersize=10,
                linestyle='None', label='Prototype'),
        Line2D([0], [0], marker='x', color='black', markersize=6,
                linestyle='None', label='Sample')
    ])

    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

    # Save 3D figure
    output_path = os.path.join(output_dir, 'prototypes_pca_3d.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"3D PCA visualization saved to: {output_path}")

    plt.show()


def compute_and_plot_distance_matrix(sample_embeddings, predicate_labels_all, predicate_labels_filtered, output_dir):
    """Compute and plot average distance matrix between sample groups and their centroids.

    For each predicate, compute the centroid of its samples (in original embedding space).
    Then compute the average Euclidean distance from each predicate's samples to each centroid.
    The resulting 5x5 matrix is plotted as a confusion-matrix-style heatmap.
    """
    # Group sample embeddings by predicate
    predicate_embeddings = {pred: [] for pred in predicate_labels_filtered}
    for sample in sample_embeddings:
        gt_label_idx = sample['gt_label']
        if gt_label_idx >= len(predicate_labels_all):
            continue
        predicate = predicate_labels_all[gt_label_idx]
        if predicate in predicate_embeddings:
            predicate_embeddings[predicate].append(sample['embedding'])

    # Convert to numpy arrays and compute centroids
    embeddings_arrays = {}
    centroids = {}
    for pred in predicate_labels_filtered:
        if predicate_embeddings[pred]:
            arr = np.array(predicate_embeddings[pred])
            embeddings_arrays[pred] = arr
            centroids[pred] = arr.mean(axis=0)

    n = len(predicate_labels_filtered)
    distance_matrix = np.zeros((n, n))

    for i, pred_i in enumerate(predicate_labels_filtered):
        if pred_i not in embeddings_arrays:
            continue
        samples_i = embeddings_arrays[pred_i]
        for j, pred_j in enumerate(predicate_labels_filtered):
            if pred_j not in centroids:
                continue
            centroid_j = centroids[pred_j]
            # Average Euclidean distance from samples of pred_i to centroid of pred_j
            dists = np.linalg.norm(samples_i - centroid_j, axis=1)
            distance_matrix[i, j] = dists.mean()

    print(f"\nDistance matrix (avg Euclidean distance: samples → centroids):")
    for i, pred_i in enumerate(predicate_labels_filtered):
        for j, pred_j in enumerate(predicate_labels_filtered):
            print(f"  samples({pred_i}) → centroid({pred_j}): {distance_matrix[i, j]:.4f}")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(distance_matrix, cmap='Blues')

    # Add text annotations
    vmax = distance_matrix.max()
    for i in range(n):
        for j in range(n):
            val = distance_matrix[i, j]
            color = 'white' if val > vmax * 0.65 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(predicate_labels_filtered, fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels(predicate_labels_filtered, fontsize=10)
    ax.set_xlabel('Centroid of', fontsize=12)
    ax.set_ylabel('Samples of', fontsize=12)
    ax.set_title('Avg Distance: Samples → Centroids\n(Original Embedding Space)',
                 fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Euclidean Distance', fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'distance_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Distance matrix saved to: {save_path}")
    plt.show()


def main():
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load embeddings data (includes both samples and prototypes)
    sample_embeddings, prototypes_data = load_embeddings_data(embeddings_path)

    # Get predicate labels
    predicate_labels_all = load_predicate_labels(config_path)

    # Check if we have prototypes data
    if prototypes_data is None:
        print("\nERROR: No prototypes found in embeddings file!")
        print("Please make sure you ran the test with a DPL model after code changes.")
        print("The embeddings file should contain both sample_embeddings and prototypes.")
        return

    # Extract prototypes and sigma
    proto_emb = prototypes_data['proto_emb']
    sigma = prototypes_data['sigma']

    # Create predicate to index mapping
    predicate_to_idx = {label: idx for idx, label in enumerate(predicate_labels_all)}

    # Filter samples to selected predicates, selecting top samples closest to prototype
    sample_embeddings = filter_samples(sample_embeddings, predicate_labels_all, proto_emb, predicate_to_idx)

    # Create filtered list for visualization
    predicate_labels_filtered = [pred for pred in SELECTED_PREDICATES if pred in predicate_labels_all]

    # Filter prototypes to only selected predicates
    selected_indices = [predicate_to_idx[pred] for pred in predicate_labels_filtered if pred in predicate_to_idx]

    proto_emb = proto_emb[selected_indices]
    sigma = sigma[selected_indices]

    print(f"\nPrototype embeddings shape: {proto_emb.shape}")
    print(f"Sigma shape: {sigma.shape}")
    print(f"Number of predicates (filtered): {len(predicate_labels_filtered)}")

    # Validate dimensions
    if len(predicate_labels_filtered) != proto_emb.shape[0]:
        print(f"\nWARNING: Number of labels ({len(predicate_labels_filtered)}) doesn't match "
              f"prototype count ({proto_emb.shape[0]})")

    # Visualize with PCA (both prototypes and samples)
    print(f"\nGenerating PCA visualization with {len(sample_embeddings)} sample points...")
    visualize_prototypes_pca(proto_emb, sigma, sample_embeddings, predicate_labels_filtered, predicate_labels_all, output_dir)

    # Create 3D visualization if requested
    if do_3d:
        print(f"\nGenerating 3D PCA visualization...")
        visualize_prototypes_3d_pca(proto_emb, sigma, sample_embeddings, predicate_labels_filtered, predicate_labels_all, output_dir)

    # Generate distance matrix heatmap
    print(f"\nGenerating distance matrix...")
    compute_and_plot_distance_matrix(sample_embeddings, predicate_labels_all, predicate_labels_filtered, output_dir)

    print("\nVisualization complete!")
    print(f"Prototypes: {len(predicate_labels_filtered)}")
    print(f"Samples: {len(sample_embeddings)}")


if __name__ == '__main__':
    main()
