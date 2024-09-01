from collections import defaultdict
from computational_model import Ciphertext


def create_mask(indices: set[int], n: int) -> list[int]:
    """Create a mask of length n with 1s at the indices specified."""
    return [1 if i in indices else 0 for i in range(n)]


def mask_and_rotate(input: Ciphertext, permutation: dict[int, int]) -> Ciphertext:
    """Compute the naive approach to permutate the data entries in an FHE ciphertext."""

    # maps a rotation amount to the indices that should be rotated by that amount
    permutation_groups = defaultdict(set)
    for i, sigma_i in permutation.items():
        permutation_groups[sigma_i - i].add(i)

    print(f"Computing the permutation with {len(permutation_groups)} groups")

    result = Ciphertext([0] * len(input))
    for shift, indices in permutation_groups.items():
        mask = create_mask(indices, len(input))
        result += (input * mask).rotate(shift)

    return result
