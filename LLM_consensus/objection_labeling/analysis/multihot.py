"""Multi-hot encoding utilities for objection dimension labels.

Canonical dimension order (7 dims = 6 scoring dimensions + Other):
Mechanical_Safety, Swelling_Performance, Endothelialization,
SMC_inhibition, Anti_inflammation, Thrombogenicity, Other
"""

DIMENSIONS = [
    "Mechanical_Safety",
    "Swelling_Performance",
    "Endothelialization",
    "SMC_inhibition",
    "Anti_inflammation",
    "Thrombogenicity",
    "Other",
]


def labels_to_vector(labels):
    """Convert a list of dimension labels into a multi-hot binary vector.

    Vector order follows DIMENSIONS. Raises on unknown labels.
    """
    label_set = set(labels)
    unknown = label_set - set(DIMENSIONS)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}")
    return [1 if dim in label_set else 0 for dim in DIMENSIONS]


def vector_to_labels(vector):
    """Convert a multi-hot binary vector back into a list of dimension labels."""
    if len(vector) != len(DIMENSIONS):
        raise ValueError(f"Vector length {len(vector)} != {len(DIMENSIONS)}")
    return [dim for dim, bit in zip(DIMENSIONS, vector) if bit]
