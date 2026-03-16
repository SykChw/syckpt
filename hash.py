"""LSH-based hash system for hyperparameters.

This implements Locality-Sensitive Hashing for hyperparameter spaces:
- Similar hyperparams produce similar hashes (or hashes in same bucket)
- Uses quantization to group similar continuous values
- Provides similarity comparison between configs
"""

import hashlib
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn


DEFAULT_HASH_FACTORS: Set[str] = {
    "lr",
    "learning_rate",
    "seed",
    "batch_size",
    "num_epochs",
    "weight_decay",
    "momentum",
    "beta1",
    "beta2",
    "eps",
}


def quantize_value(value: float, scales: List[float] = None) -> float:
    """Quantize a continuous value to nearest scale.

    For learning rates: maps to nearest value in [0.001, 0.01, 0.1, 1.0]
    This ensures lr=0.009 and lr=0.011 both map to 0.01
    """
    if scales is None:
        if value == 0:
            return 0
        # Log-scale quantization for learning rates
        magnitude = abs(value)
        sign = 1 if value > 0 else -1

        # Common learning rate scales
        scales = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]

        closest = min(scales, key=lambda s: abs(magnitude - s))
        return sign * closest

    return min(scales, key=lambda s: abs(value - s))


def quantize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Quantize all numeric values in a dict."""
    result = {}
    for key, value in data.items():
        if isinstance(value, float):
            result[key] = quantize_value(value)
        elif isinstance(value, dict):
            result[key] = quantize_dict(value)
        elif isinstance(value, list):
            result[key] = [
                quantize_value(v) if isinstance(v, float) else v for v in value
            ]
        else:
            result[key] = value
    return result


class LSHHashGenerator:
    """Locality-Sensitive Hash generator for hyperparameters.

    Key properties:
    - Similar configs produce similar/hamming-close hashes
    - Uses quantization to group similar continuous values
    - Provides bucketing for finding similar experiments

    Example:
        >>> gen = LSHHashGenerator()
        >>> h1 = gen.generate({"lr": 0.01, "batch_size": 32})
        >>> h2 = gen.generate({"lr": 0.011, "batch_size": 32})
        >>> # h1 and h2 are different but "close" (hamming distance small)
        >>> gen.similarity(h1, h2)  # High similarity!
    """

    def __init__(
        self,
        hash_length: int = 8,
        num_bands: int = 4,
        factors: Optional[Set[str]] = None,
    ):
        """Initialize LSH hash generator.

        Args:
            hash_length: Length of hash string to generate
            num_bands: Number of LSH bands (more = finer locality)
            factors: Which config keys to include in hash
        """
        self.hash_length = hash_length
        self.num_bands = num_bands
        self.factors = factors or DEFAULT_HASH_FACTORS

        # Generate random projection matrices for LSH
        # Each band uses different random projection
        np.random.seed(42)  # Fixed seed for reproducibility
        self._projection_matrices = [
            np.random.randn(16, len(self.factors)) for _ in range(num_bands)
        ]

    def _get_factor_vector(self, config: Dict[str, Any]) -> np.ndarray:
        """Extract and normalize factor values into a vector."""
        sorted_factors = sorted(self.factors)
        values = []

        for factor in sorted_factors:
            value = config.get(factor, 0)
            if isinstance(value, (int, float)):
                values.append(float(value))
            else:
                values.append(hash(str(value)) % 1000 / 100.0)

        arr = np.array(values, dtype=np.float32)

        # Normalize to unit sphere
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm

        return arr

    def _compute_band_hashes(self, vector: np.ndarray) -> List[int]:
        """Compute band hashes using random projections."""
        band_hashes = []

        for matrix in self._projection_matrices:
            # Project onto random hyperplane
            projected = np.dot(matrix, vector)
            # Binary hash: positive = 1, negative = 0
            band_hash = int("".join(["1" if p > 0 else "0" for p in projected]), 2)
            band_hashes.append(band_hash)

        return band_hashes

    def generate(self, config: Dict[str, Any]) -> str:
        """Generate LSH hash from config.

        Uses quantization + random projections for locality sensitivity.
        Similar configs will have similar (low hamming distance) hashes.
        """
        # Step 1: Quantize continuous values
        quantized = quantize_dict(config)

        # Step 2: Get factor vector
        vector = self._get_factor_vector(quantized)

        # Step 3: Compute band hashes (for LSH bucketing)
        band_hashes = self._compute_band_hashes(vector)

        # Step 4: Create final hash from band hashes
        combined = "".join(str(h) for h in band_hashes)
        full_hash = hashlib.sha256(combined.encode()).hexdigest()

        return full_hash[: self.hash_length]

    def generate_from_components(
        self,
        config: Dict[str, Any],
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> str:
        """Generate hash from config + model + optimizer."""
        # Add model signature
        if model is not None:
            num_params = sum(p.numel() for p in model.parameters())
            config["_num_params"] = num_params

            # Add key layer types
            layer_types = []
            for m in model.modules():
                if len(list(m.children())) == 0:
                    layer_types.append(type(m).__name__)
            config["_layers"] = "_".join(sorted(set(layer_types))[:5])

        # Add optimizer signature
        if optimizer is not None:
            opt_type = type(optimizer).__name__
            config["_opt_type"] = opt_type

            if optimizer.param_groups:
                pg = optimizer.param_groups[0]
                config["_opt_lr"] = pg.get("lr", 0)
                config["_opt_momentum"] = pg.get("momentum", 0)

        return self.generate(config)

    def get_bucket(self, config: Dict[str, Any]) -> Tuple[int, ...]:
        """Get LSH bucket for a config.

        Returns tuple of band hashes - configs in same bucket
        have collided in at least one band.
        """
        quantized = quantize_dict(config)
        vector = self._get_factor_vector(quantized)
        return tuple(self._compute_band_hashes(vector))

    def similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two hashes (0 to 1).

        Uses hamming distance on binary representations.
        """
        # Convert hex to binary
        bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)

        # Pad to same length
        max_len = max(len(bin1), len(bin2))
        bin1 = bin1.zfill(max_len)
        bin2 = bin2.zfill(max_len)

        # Calculate hamming similarity
        matches = sum(c1 == c2 for c1, c2 in zip(bin1, bin2))
        return matches / max_len

    def find_similar_configs(
        self,
        config: Dict[str, Any],
        existing_configs: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Find most similar configs from a list.

        Returns list of (config, similarity) tuples sorted by similarity.
        """
        target_hash = self.generate(config)

        similarities = []
        for existing in existing_configs:
            existing_hash = self.generate(existing)
            sim = self.similarity(target_hash, existing_hash)
            similarities.append((existing, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# Backwards compatibility alias
class HashGenerator(LSHHashGenerator):
    """Alias for backwards compatibility."""

    pass
