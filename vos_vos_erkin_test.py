from typing import Optional
from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis.strategies import composite, integers, permutations

from computational_model import Ciphertext
from vos_vos_erkin import (
    vos_vos_erkin,
    implement_shift_network,
    Mapping,
    default_shift_order,
)


@dataclass(frozen=True)
class TestCase:
    __test__ = False
    # The number of slots of each ciphertext
    ciphertext_size: int
    # The expected number of rotation groups
    expected_num_groups: int
    # The mapping to implement
    mapping: Mapping
    # The number of ciphertexts
    num_ciphertexts: int
    # The order of power-of-two shifts to use
    shift_order: Optional[list[int]] = None


FIG3 = TestCase(
    mapping=[
        ((0, 0), (0, 13)),
        ((0, 1), (0, 8)),
        ((0, 2), (0, 4)),
        ((0, 3), (0, 0)),
        ((0, 4), (0, 11)),
        ((0, 5), (0, 7)),
        ((0, 6), (0, 14)),
        ((0, 7), (0, 5)),
        ((0, 8), (0, 15)),
        ((0, 9), (0, 3)),
        ((0, 10), (0, 12)),
        ((0, 11), (0, 6)),
        ((0, 12), (0, 10)),
        ((0, 13), (0, 2)),
        ((0, 14), (0, 9)),
        ((0, 15), (0, 1)),
    ],
    ciphertext_size=16,
    num_ciphertexts=1,
    expected_num_groups=3,
)


# replicate a single slot to all slots in a single ciphertext
FULL_REPLICATION = TestCase(
    mapping=[
        ((0, 0), (0, 0)),
        ((0, 0), (0, 1)),
        ((0, 0), (0, 2)),
        ((0, 0), (0, 3)),
        ((0, 0), (0, 4)),
        ((0, 0), (0, 5)),
        ((0, 0), (0, 6)),
        ((0, 0), (0, 7)),
        ((0, 0), (0, 8)),
        ((0, 0), (0, 9)),
        ((0, 0), (0, 10)),
        ((0, 0), (0, 11)),
        ((0, 0), (0, 12)),
        ((0, 0), (0, 13)),
        ((0, 0), (0, 14)),
        ((0, 0), (0, 15)),
    ],
    ciphertext_size=16,
    num_ciphertexts=1,
    expected_num_groups=1,
)


# replicate the first slot to the first half of one ciphertext,
# the second slot to the second half
TWO_REPLICATION = TestCase(
    mapping=[
        ((0, 0), (0, 0)),
        ((0, 0), (0, 1)),
        ((0, 0), (0, 2)),
        ((0, 0), (0, 3)),
        ((0, 0), (0, 4)),
        ((0, 0), (0, 5)),
        ((0, 0), (0, 6)),
        ((0, 0), (0, 7)),
        ((0, 1), (0, 8)),
        ((0, 1), (0, 9)),
        ((0, 1), (0, 10)),
        ((0, 1), (0, 11)),
        ((0, 1), (0, 12)),
        ((0, 1), (0, 13)),
        ((0, 1), (0, 14)),
        ((0, 1), (0, 15)),
    ],
    ciphertext_size=16,
    num_ciphertexts=1,
    expected_num_groups=2,
)

TWO_REPLICATION_ALTERNATE_SHIFT_ORDER = TestCase(
    mapping=[
        ((0, 0), (0, 0)),
        ((0, 0), (0, 1)),
        ((0, 0), (0, 2)),
        ((0, 0), (0, 3)),
        ((0, 0), (0, 4)),
        ((0, 0), (0, 5)),
        ((0, 0), (0, 6)),
        ((0, 0), (0, 7)),
        ((0, 1), (0, 8)),
        ((0, 1), (0, 9)),
        ((0, 1), (0, 10)),
        ((0, 1), (0, 11)),
        ((0, 1), (0, 12)),
        ((0, 1), (0, 13)),
        ((0, 1), (0, 14)),
        ((0, 1), (0, 15)),
    ],
    ciphertext_size=16,
    num_ciphertexts=1,
    expected_num_groups=1,
    shift_order=[8, 4, 2, 1],
)


# Swap two ciphertexts (shouldn't require any rotations)
SWAP_TWO_CIPHERTEXTS = TestCase(
    mapping=[
        # ct 0 -> ct 1
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((0, 2), (1, 2)),
        ((0, 3), (1, 3)),
        # ct 1 -> ct 0
        ((1, 0), (0, 0)),
        ((1, 1), (0, 1)),
        ((1, 2), (0, 2)),
        ((1, 3), (0, 3)),
    ],
    ciphertext_size=4,
    num_ciphertexts=2,
    expected_num_groups=1,
)


REORDER_THREE_CIPHERTEXTS = TestCase(
    mapping=[
        # ct 0 -> ct 2
        ((0, 0), (2, 0)),
        ((0, 1), (2, 1)),
        ((0, 2), (2, 2)),
        ((0, 3), (2, 3)),
        # ct 1 -> ct 0
        ((1, 0), (0, 0)),
        ((1, 1), (0, 1)),
        ((1, 2), (0, 2)),
        ((1, 3), (0, 3)),
        # ct 2 -> ct 1
        ((2, 0), (1, 0)),
        ((2, 1), (1, 1)),
        ((2, 2), (1, 2)),
        ((2, 3), (1, 3)),
    ],
    ciphertext_size=4,
    num_ciphertexts=3,
    expected_num_groups=1,
)


# rotate only by one, so that the ciphertext splits
SINGLE_ROT_SPLIT = TestCase(
    mapping=[
        # ct 0
        ((0, 0), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 2), (0, 3)),
        ((0, 3), (1, 0)),
        # ct 1
        ((1, 0), (1, 1)),
        ((1, 1), (1, 2)),
        ((1, 2), (1, 3)),
        ((1, 3), (2, 0)),
        # ct 2
        ((2, 0), (2, 1)),
        ((2, 1), (2, 2)),
        ((2, 2), (2, 3)),
        ((2, 3), (0, 0)),
    ],
    ciphertext_size=4,
    num_ciphertexts=3,
    expected_num_groups=1,
)


@pytest.mark.parametrize(
    "test_case",
    [
        FIG3,
        FULL_REPLICATION,
        TWO_REPLICATION,
        TWO_REPLICATION_ALTERNATE_SHIFT_ORDER,
        SWAP_TWO_CIPHERTEXTS,
        REORDER_THREE_CIPHERTEXTS,
        SINGLE_ROT_SPLIT,
    ],
)
def test_network_implementation(test_case):
    num_ciphertexts = test_case.num_ciphertexts
    ciphertext_size = test_case.ciphertext_size
    mapping = test_case.mapping
    shift_order = test_case.shift_order

    rot_groups = vos_vos_erkin(
        num_ciphertexts, ciphertext_size, mapping, shift_order=test_case.shift_order
    )

    # allow zero for property tests that don't know the expected group count in
    # advance
    if test_case.expected_num_groups > 0:
        assert len(rot_groups) == test_case.expected_num_groups

    # example is integers from 1...num_ciphertexts*ciphertext_size
    # in row-major order
    input = []
    for i in range(num_ciphertexts):
        input.append(
            Ciphertext([1 + j + i * ciphertext_size for j in range(ciphertext_size)])
        )

    # print(f"{input=}")

    output = implement_shift_network(
        input, mapping, rot_groups, shift_order=shift_order
    )

    expected = []
    for _ in range(num_ciphertexts):
        expected.append(Ciphertext([0] * ciphertext_size))
    for source, target in mapping:
        source_ct, source_slot = source
        target_ct, target_slot = target
        expected[target_ct].data[target_slot] = input[source_ct].data[source_slot]

    assert output == expected


@composite
def random_testcase(draw, min_ciphertexts=1, max_ciphertexts=32, ciphertext_size=8):
    """Generate a random set of matchups."""
    num_ciphertexts = draw(
        integers(min_value=min_ciphertexts, max_value=max_ciphertexts)
    )
    shifts = default_shift_order(ciphertext_size * num_ciphertexts)
    shift_order = draw(permutations(shifts))

    # for each target slot, provide a random source slot
    mapping = []
    for ct in range(num_ciphertexts):
        for slot in range(ciphertext_size):
            source_ct = draw(integers(min_value=0, max_value=num_ciphertexts - 1))
            source_slot = draw(integers(min_value=0, max_value=ciphertext_size - 1))
            mapping.append(((source_ct, source_slot), (ct, slot)))

    return TestCase(
        ciphertext_size=ciphertext_size,
        expected_num_groups=0,  # unknown
        mapping=mapping,
        num_ciphertexts=num_ciphertexts,
        shift_order=shift_order,
    )


@settings(deadline=100000, max_examples=75)
@given(random_testcase())
def test_random_multiciphertext_mapping(test_case):
    print(test_case.mapping)
    test_network_implementation(test_case)
