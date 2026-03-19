#!/usr/bin/env python3
"""
Script to visualize DPL prototypes and samples using PCA visualization.

This script loads previously extracted prototype embeddings and sample embeddings
from the pkl file, and creates PCA visualizations with sigma uncertainty circles.
No need to load the model - everything is already saved in the pkl file.
"""

import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from pysgg.data import get_dataset_statistics
from pysgg.config import cfg

# Configuration
embeddings_path = "/root/autodl-tmp/visualize/relation_embeddings.pkl"
config_path = "/root/autodl-tmp/ckpt/config.yml"
output_dir = "/root/autodl-tmp/visualize"
do_3d = False


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


def visualize_prototypes_pca(proto_emb, sigma, sample_embeddings, predicate_labels, output_dir, n_components=2):
    """Visualize prototypes and samples using PCA with sigma uncertainty circles."""

    # Normalize prototypes
    proto_norm = proto_emb / np.linalg.norm(proto_emb, axis=1, keepdims=True)

    # Collect sample embeddings
    sample_vecs = np.array([item['embedding'] for item in sample_embeddings])
    sample_labels = [item['gt_label'] for item in sample_embeddings]

    # Normalize sample embeddings
    sample_norm = sample_vecs / np.linalg.norm(sample_vecs, axis=1, keepdims=True)

    # Apply PCA
    print(f"\nApplying PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    proto_pca = pca.fit_transform(proto_norm)
    sample_pca = pca.transform(sample_norm)

    # Transform sigma to PCA space
    # For each dimension, compute the variance in PCA space
    # We'll use the average sigma as the radius
    avg_sigma = np.mean(sigma, axis=1)  # (num_predicates,)
    sigma_pca = avg_sigma  # Use average sigma as radius

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Create visualization
    plt.figure(figsize=(16, 12))

    # Use different colors for different predicates
    n_predicates = len(predicate_labels)

    # Generate distinct colors
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, min(n_predicates, 20)))

    # Plot samples first (smaller points)
    for i, (label_idx, point) in enumerate(zip(sample_labels, sample_pca)):
        if label_idx >= len(predicate_labels):
            continue
        color_idx = label_idx % 20
        plt.scatter(point[0], point[1], c=[colors[color_idx]], s=15,
                  edgecolors='black', linewidth=0.3, alpha=0.4,
                  marker='x', zorder=5)  # Draw under prototypes

    # Plot each predicate prototype (larger points) with sigma circles
    for i, (label, point) in enumerate(zip(predicate_labels, proto_pca)):
        color_idx = i % 20

        # Draw sigma uncertainty circle (dashed line)
        radius = sigma_pca[i] * 50  # Scale factor for visualization
        circle = Circle((point[0], point[1]), radius=radius,
                      edgecolor=colors[color_idx], facecolor=colors[color_idx],
                      linewidth=1.5, linestyle='--', alpha=0.3, zorder=8)
        plt.gca().add_patch(circle)

        # Plot prototype point
        plt.scatter(point[0], point[1], c=[colors[color_idx]], s=250,
                  edgecolors='black', linewidth=1.5, alpha=0.9,
                  marker='o', label=label if i == 0 else "",
                  zorder=10)  # Draw on top

        # Add label for each prototype point
        plt.annotate(label, (point[0], point[1]),
                   fontsize=10, ha='center', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[color_idx], alpha=0.5),
                   zorder=11)  # Draw on top

    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='black', markersize=10,
                linestyle='None', label='Prototypes'),
        Line2D([0], [0], marker='x', color='black', markersize=6,
                linestyle='None', label='Samples'),
        plt.Line2D([0], [0], linestyle='--', color='black',
                   label='Sigma uncertainty')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=12)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    plt.title('DPL Prototypes and Samples with Sigma Uncertainty - PCA Visualization',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(output_dir, 'prototypes_pca_with_sigma.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"PCA visualization with sigma saved to: {save_path}")

    # Save PCA data
    data_save_path = os.path.join(output_dir, 'prototypes_pca_data.pkl')
    pca_data = {
        'proto_pca': proto_pca,
        'sample_pca': sample_pca,
        'sigma_pca': sigma_pca,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'predicate_labels': predicate_labels,
        'proto_emb': proto_emb,
        'sigma': sigma
    }
    with open(data_save_path, 'wb') as f:
        pickle.dump(pca_data, f)
    print(f"PCA data saved to: {data_save_path}")

    plt.show()


def visualize_prototypes_3d_pca(proto_emb, sigma, predicate_labels, output_dir):
    """Create 3D PCA visualization with sigma uncertainty spheres."""

    # Normalize prototypes
    proto_norm = proto_emb / np.linalg.norm(proto_emb, axis=1, keepdims=True)

    # Apply 3D PCA
    print(f"\nApplying 3D PCA...")
    pca = PCA(n_components=3)
    proto_pca = pca.fit_transform(proto_norm)

    # Get average sigma
    avg_sigma = np.mean(sigma, axis=1)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Create 3D visualization
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')

    n_predicates = len(predicate_labels)
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, min(n_predicates, 20)))

    # Plot each predicate
    for i, (label, point) in enumerate(zip(predicate_labels, proto_pca)):
        color_idx = i % 20

        # Draw wireframe sphere for sigma uncertainty
        radius = avg_sigma[i] * 50  # Scale factor for visualization
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = radius * np.outer(np.cos(u), np.sin(v)) + point[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + point[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + point[2]

        ax.plot_wireframe(x, y, z, color=colors[color_idx], alpha=0.2, linewidth=0.5)

        # Plot prototype point
        ax.scatter(point[0], point[1], point[2], c=[colors[color_idx]], s=150,
                 edgecolors='black', linewidth=1, alpha=0.9)

        # Add label
        ax.text(point[0], point[1], point[2], label, fontsize=7, ha='center')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=12)
    ax.set_title('DPL Prototype Embeddings with Sigma Uncertainty - 3D PCA Visualization',
                 fontsize=14, fontweight='bold')

    # Save 3D figure
    output_path = os.path.join(output_dir, 'prototypes_pca_3d_with_sigma.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"3D PCA visualization with sigma saved to: {output_path}")

    plt.show()


def main():
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load embeddings data (includes both samples and prototypes)
    sample_embeddings, prototypes_data = load_embeddings_data(embeddings_path)

    # Get predicate labels
    predicate_labels = load_predicate_labels(config_path)

    # Check if we have prototypes data
    if prototypes_data is None:
        print("\nERROR: No prototypes found in embeddings file!")
        print("Please make sure you ran the test with a DPL model after the code changes.")
        print("The embeddings file should contain both sample_embeddings and prototypes.")
        return

    # Extract prototypes and sigma
    proto_emb = prototypes_data['proto_emb']
    sigma = prototypes_data['sigma']

    print(f"\nPrototype embeddings shape: {proto_emb.shape}")
    print(f"Sigma shape: {sigma.shape}")
    print(f"Number of predicates: {len(predicate_labels)}")

    # Validate dimensions
    if len(predicate_labels) != proto_emb.shape[0]:
        print(f"\nWARNING: Number of labels ({len(predicate_labels)}) doesn't match "
              f"prototype count ({proto_emb.shape[0]})")

    # Visualize with PCA (both prototypes and samples)
    print(f"\nGenerating PCA visualization with {len(sample_embeddings)} sample points...")
    visualize_prototypes_pca(proto_emb, sigma, sample_embeddings, predicate_labels, output_dir)

    # Create 3D visualization if requested
    if do_3d:
        print(f"\nGenerating 3D PCA visualization...")
        visualize_prototypes_3d_pca(proto_emb, sigma, predicate_labels, output_dir)

    print("\nVisualization complete!")
    print(f"Prototypes: {len(predicate_labels)}")
    print(f"Samples: {len(sample_embeddings)}")


if __name__ == '__main__':
    main()
