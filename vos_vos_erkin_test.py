from typing import Optional
from dataclasses import dataclass

import pytest
from hypothesis import given, settings, example
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


cpp_test_template = """TEST(ImplementShiftNetworkTest, Test{name}) {{
  int64_t numCts = {num_ciphertexts};
  int64_t ctSize = {ciphertext_size};
  Mapping mapping;
{add_slots}
  VosVosErkinShiftNetworks shiftNetworks;
  auto scheme = shiftNetworks.findShiftScheme(mapping);
  EXPECT_EQ(scheme.rotationGroups.size(), {expected_num_groups});
  simulateShiftNetwork(mapping, scheme, numCts, ctSize);
}}
"""


randomized_cpp_test_template = """TEST(ImplementShiftNetworkTest, Test{name}) {{
  int64_t numCts = {num_ciphertexts};
  int64_t ctSize = {ciphertext_size};
  Mapping mapping(ctSize, numCts);
{add_slots}
  VosVosErkinShiftNetworks shiftNetworks;
  auto scheme = shiftNetworks.findShiftScheme(mapping);
  simulateShiftNetwork(mapping, scheme, numCts, ctSize);
}}
"""

add_slot_template = "  mapping.add(CtSlot({source_ct}, {source_slot}), CtSlot({target_ct}, {target_slot}));"


def generate_cpp_test_case(name: str, test_case: TestCase, random:bool = False) -> str:
    add_slots = []
    for source, target in test_case.mapping:
        source_ct, source_slot = source
        target_ct, target_slot = target
        add_slots.append(
            add_slot_template.format(
                source_ct=source_ct,
                source_slot=source_slot,
                target_ct=target_ct,
                target_slot=target_slot,
            )
        )
    if random:
        return randomized_cpp_test_template.format(
            name=name,
            num_ciphertexts=test_case.num_ciphertexts,
            ciphertext_size=test_case.ciphertext_size,
            add_slots="\n".join(add_slots),
        )

    return cpp_test_template.format(
        name=name,
        num_ciphertexts=test_case.num_ciphertexts,
        ciphertext_size=test_case.ciphertext_size,
        add_slots="\n".join(add_slots),
        expected_num_groups=test_case.expected_num_groups,
    )


TEST_CASES = {
    "Fig3": FIG3,
    "FullReplication": FULL_REPLICATION,
    "TwoReplication": TWO_REPLICATION,
    "TwoReplicationAlternateShiftOrder": TWO_REPLICATION_ALTERNATE_SHIFT_ORDER,
    "SwapTwoCiphertexts": SWAP_TWO_CIPHERTEXTS,
    "ReorderThreeCiphertexts": REORDER_THREE_CIPHERTEXTS,
    "SingleRotSplit": SINGLE_ROT_SPLIT,
}


# for name, test_case in TEST_CASES.items():
#     print(generate_cpp_test_case(name, test_case))


@pytest.mark.parametrize("test_case", TEST_CASES.values())
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


# One of the resulting rotation groups consists of a single source is one
# ciphertext that remains fixed the whole time. In this case the other
# ciphertext must be treated as zero.
TWO_CT_ONE_UNTOUCHED = TestCase(
    ciphertext_size=8,
    expected_num_groups=0,
    mapping=[
        ((0, 0), (0, 0)),
        ((0, 0), (0, 1)),
        ((0, 0), (0, 2)),
        ((0, 0), (0, 3)),
        ((0, 0), (0, 4)),
        ((0, 0), (0, 5)),
        ((0, 0), (0, 6)),
        ((0, 0), (0, 7)),
        ((0, 0), (1, 0)),
        ((0, 0), (1, 1)),
        ((1, 2), (1, 2)),
        ((0, 0), (1, 3)),
        ((0, 0), (1, 4)),
        ((0, 0), (1, 5)),
        ((0, 7), (1, 6)),
        ((0, 0), (1, 7)),
    ],
    num_ciphertexts=2,
    shift_order=[1, 2, 4, 8],
)

HARD_EXAMPLE_1 = TestCase(
    ciphertext_size=8,
    expected_num_groups=0,
    mapping=[
        ((0, 0), (0, 0)),
        ((12, 0), (0, 1)),
        ((23, 0), (0, 2)),
        ((22, 0), (0, 3)),
        ((0, 0), (0, 4)),
        ((8, 5), (0, 5)),
        ((0, 3), (0, 6)),
        ((0, 0), (0, 7)),
        ((18, 3), (1, 0)),
        ((20, 0), (1, 1)),
        ((15, 2), (1, 2)),
        ((23, 0), (1, 3)),
        ((13, 0), (1, 4)),
        ((20, 0), (1, 5)),
        ((23, 7), (1, 6)),
        ((0, 0), (1, 7)),
        ((0, 0), (2, 0)),
        ((16, 6), (2, 1)),
        ((20, 0), (2, 2)),
        ((0, 0), (2, 3)),
        ((15, 0), (2, 4)),
        ((0, 0), (2, 5)),
        ((11, 0), (2, 6)),
        ((0, 0), (2, 7)),
        ((13, 2), (3, 0)),
        ((0, 0), (3, 1)),
        ((0, 0), (3, 2)),
        ((0, 0), (3, 3)),
        ((22, 0), (3, 4)),
        ((0, 0), (3, 5)),
        ((0, 0), (3, 6)),
        ((0, 0), (3, 7)),
        ((20, 0), (4, 0)),
        ((19, 0), (4, 1)),
        ((0, 0), (4, 2)),
        ((0, 0), (4, 3)),
        ((11, 0), (4, 4)),
        ((0, 0), (4, 5)),
        ((13, 0), (4, 6)),
        ((0, 0), (4, 7)),
        ((16, 1), (5, 0)),
        ((23, 0), (5, 1)),
        ((21, 0), (5, 2)),
        ((12, 0), (5, 3)),
        ((15, 0), (5, 4)),
        ((0, 0), (5, 5)),
        ((0, 0), (5, 6)),
        ((0, 0), (5, 7)),
        ((21, 0), (6, 0)),
        ((0, 0), (6, 1)),
        ((22, 5), (6, 2)),
        ((15, 0), (6, 3)),
        ((19, 0), (6, 4)),
        ((24, 0), (6, 5)),
        ((0, 0), (6, 6)),
        ((0, 0), (6, 7)),
        ((18, 0), (7, 0)),
        ((15, 0), (7, 1)),
        ((0, 0), (7, 2)),
        ((0, 0), (7, 3)),
        ((12, 0), (7, 4)),
        ((0, 0), (7, 5)),
        ((13, 6), (7, 6)),
        ((13, 0), (7, 7)),
        ((13, 6), (8, 0)),
        ((0, 0), (8, 1)),
        ((0, 0), (8, 2)),
        ((0, 0), (8, 3)),
        ((18, 0), (8, 4)),
        ((0, 0), (8, 5)),
        ((23, 0), (8, 6)),
        ((24, 0), (8, 7)),
        ((18, 0), (9, 0)),
        ((0, 0), (9, 1)),
        ((13, 0), (9, 2)),
        ((1, 0), (9, 3)),
        ((18, 0), (9, 4)),
        ((0, 0), (9, 5)),
        ((11, 0), (9, 6)),
        ((0, 0), (9, 7)),
        ((0, 0), (10, 0)),
        ((0, 0), (10, 1)),
        ((23, 0), (10, 2)),
        ((0, 0), (10, 3)),
        ((0, 0), (10, 4)),
        ((11, 0), (10, 5)),
        ((12, 0), (10, 6)),
        ((15, 0), (10, 7)),
        ((15, 2), (11, 0)),
        ((0, 0), (11, 1)),
        ((0, 0), (11, 2)),
        ((0, 0), (11, 3)),
        ((0, 0), (11, 4)),
        ((24, 0), (11, 5)),
        ((0, 0), (11, 6)),
        ((0, 0), (11, 7)),
        ((0, 0), (12, 0)),
        ((16, 4), (12, 1)),
        ((18, 0), (12, 2)),
        ((0, 0), (12, 3)),
        ((0, 0), (12, 4)),
        ((0, 0), (12, 5)),
        ((0, 0), (12, 6)),
        ((11, 0), (12, 7)),
        ((0, 0), (13, 0)),
        ((23, 0), (13, 1)),
        ((0, 0), (13, 2)),
        ((0, 0), (13, 3)),
        ((13, 0), (13, 4)),
        ((23, 0), (13, 5)),
        ((19, 0), (13, 6)),
        ((22, 7), (13, 7)),
        ((13, 0), (14, 0)),
        ((20, 0), (14, 1)),
        ((0, 0), (14, 2)),
        ((19, 0), (14, 3)),
        ((12, 0), (14, 4)),
        ((0, 0), (14, 5)),
        ((0, 0), (14, 6)),
        ((0, 0), (14, 7)),
        ((0, 0), (15, 0)),
        ((19, 0), (15, 1)),
        ((18, 0), (15, 2)),
        ((21, 0), (15, 3)),
        ((18, 0), (15, 4)),
        ((12, 0), (15, 5)),
        ((22, 0), (15, 6)),
        ((16, 1), (15, 7)),
        ((16, 0), (16, 0)),
        ((0, 0), (16, 1)),
        ((0, 0), (16, 2)),
        ((10, 0), (16, 3)),
        ((0, 0), (16, 4)),
        ((0, 0), (16, 5)),
        ((18, 0), (16, 6)),
        ((12, 0), (16, 7)),
        ((0, 0), (17, 0)),
        ((19, 0), (17, 1)),
        ((18, 0), (17, 2)),
        ((0, 0), (17, 3)),
        ((0, 0), (17, 4)),
        ((11, 0), (17, 5)),
        ((20, 0), (17, 6)),
        ((0, 0), (17, 7)),
        ((0, 0), (18, 0)),
        ((23, 0), (18, 1)),
        ((20, 0), (18, 2)),
        ((15, 0), (18, 3)),
        ((18, 0), (18, 4)),
        ((15, 0), (18, 5)),
        ((24, 0), (18, 6)),
        ((0, 0), (18, 7)),
        ((0, 0), (19, 0)),
        ((16, 2), (19, 1)),
        ((0, 0), (19, 2)),
        ((22, 0), (19, 3)),
        ((0, 0), (19, 4)),
        ((19, 0), (19, 5)),
        ((0, 0), (19, 6)),
        ((0, 0), (19, 7)),
        ((0, 0), (20, 0)),
        ((15, 6), (20, 1)),
        ((22, 0), (20, 2)),
        ((14, 0), (20, 3)),
        ((16, 3), (20, 4)),
        ((0, 3), (20, 5)),
        ((19, 7), (20, 6)),
        ((5, 0), (20, 7)),
        ((18, 0), (21, 0)),
        ((4, 5), (21, 1)),
        ((20, 2), (21, 2)),
        ((11, 5), (21, 3)),
        ((24, 1), (21, 4)),
        ((5, 1), (21, 5)),
        ((14, 6), (21, 6)),
        ((7, 0), (21, 7)),
        ((21, 5), (22, 0)),
        ((0, 3), (22, 1)),
        ((4, 1), (22, 2)),
        ((24, 1), (22, 3)),
        ((17, 3), (22, 4)),
        ((9, 2), (22, 5)),
        ((20, 5), (22, 6)),
        ((20, 2), (22, 7)),
        ((0, 5), (23, 0)),
        ((4, 7), (23, 1)),
        ((16, 5), (23, 2)),
        ((8, 6), (23, 3)),
        ((21, 6), (23, 4)),
        ((12, 7), (23, 5)),
        ((17, 4), (23, 6)),
        ((16, 7), (23, 7)),
        ((3, 3), (24, 0)),
        ((2, 6), (24, 1)),
        ((19, 6), (24, 2)),
        ((24, 6), (24, 3)),
        ((13, 3), (24, 4)),
        ((20, 0), (24, 5)),
        ((3, 7), (24, 6)),
        ((12, 3), (24, 7)),
    ],
    num_ciphertexts=25,
    shift_order=[2, 16, 4, 64, 32, 1, 8, 128],
)


random_index = 0

@settings(deadline=100000, max_examples=75)
@given(random_testcase())
@example(TWO_CT_ONE_UNTOUCHED)
@example(HARD_EXAMPLE_1)
def test_random_multiciphertext_mapping(test_case):
    global random_index
    print(generate_cpp_test_case(f"RANDOM_{random_index}", test_case, random=True))
    random_index += 1
    test_network_implementation(test_case)
