import numpy as np

from src.analysis.encodec_latents import compute_pca, label_separability


def test_compute_pca_returns_requested_components():
    latents = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
        ]
    )

    result = compute_pca(latents, n_components=2)

    assert result.coordinates.shape == (4, 2)
    assert result.components.shape == (2, 3)
    assert result.explained_variance[0] > 0


def test_label_separability_is_high_for_clustered_labels():
    coordinates = np.array(
        [
            [-1.0, 0.0],
            [-0.9, 0.1],
            [1.0, 0.0],
            [0.9, -0.1],
        ]
    )
    labels = ["speaker_a", "speaker_a", "speaker_b", "speaker_b"]

    score = label_separability(coordinates, labels)

    assert score > 0.9
