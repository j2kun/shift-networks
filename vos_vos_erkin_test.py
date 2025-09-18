from computational_model import Ciphertext
from vos_vos_erkin import vos_vos_erkin, implement_shift_network


FIG3_MAPPING = [
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
]

# replicate a single slot to all slots in a single ciphertext
FULL_REPLICATION_MAPPING = [
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
]

# replicate the first slot to the first half of one ciphertext,
# the second slot to the second half
TWO_REPLICATION_MAPPING = [
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
]


# Swap two ciphertexts (shouldn't require any rotations)
SWAP_TWO_CIPHERTEXTS = [
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
]

REORDER_THREE_CIPHERTEXTS = [
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
]



def run_network_implementation(
    num_ciphertexts, ciphertext_size, mapping, shift_order=None
):
    rot_groups = vos_vos_erkin(
        num_ciphertexts, ciphertext_size, mapping, shift_order=shift_order
    )
    # example is integers from 1...num_ciphertexts*ciphertext_size
    # in row-major order
    input = []
    for i in range(num_ciphertexts):
        input.append(Ciphertext([j + i * ciphertext_size for j in range(ciphertext_size)]))

    print(f"{input=}")

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


def test_fig3():
    ciphertext_size = 16
    num_ciphertexts = 1
    actual = vos_vos_erkin(num_ciphertexts, ciphertext_size, FIG3_MAPPING)
    assert len(actual) == 3
    for bad_edge in [
        ((0, 0), (0, 14)),
        ((0, 0), (0, 15)),
        ((0, 1), (0, 2)),
        ((0, 1), (0, 3)),
        ((0, 2), (0, 3)),
        ((0, 4), (0, 5)),
        ((0, 4), (0, 9)),
        ((0, 8), (0, 9)),
        ((0, 11), (0, 12)),
        ((0, 11), (0, 13)),
        ((0, 12), (0, 13)),
        ((0, 14), (0, 15)),
    ]:
        assert not any(
            bad_edge[0] in group.sources and bad_edge[1] in group.sources
            for group in actual
        )
    run_network_implementation(num_ciphertexts, ciphertext_size, FIG3_MAPPING)


def test_mapping():
    ciphertext_size = 16
    num_ciphertexts = 1
    actual = vos_vos_erkin(num_ciphertexts, ciphertext_size, FULL_REPLICATION_MAPPING)
    assert len(actual) == 1
    run_network_implementation(
        num_ciphertexts, ciphertext_size, FULL_REPLICATION_MAPPING
    )


def test_mapping_2():
    ciphertext_size = 16
    num_ciphertexts = 1
    actual = vos_vos_erkin(num_ciphertexts, ciphertext_size, TWO_REPLICATION_MAPPING)
    # the default ordering of shifts creates the conflict
    assert len(actual) == 2
    run_network_implementation(
        num_ciphertexts, ciphertext_size, TWO_REPLICATION_MAPPING
    )


def test_mapping_2_with_different_ordering():
    ciphertext_size = 16
    num_ciphertexts = 1
    actual = vos_vos_erkin(
        num_ciphertexts,
        ciphertext_size,
        TWO_REPLICATION_MAPPING,
        shift_order=[8, 4, 2, 1],
    )
    # Putting 8 first allows the initial value in slot 1 to be shifted
    # away from the conflict first.
    assert len(actual) == 1
    run_network_implementation(
        num_ciphertexts,
        ciphertext_size,
        TWO_REPLICATION_MAPPING,
        shift_order=[8, 4, 2, 1],
    )


def test_swapping_two_ciphertexts():
    # this tests the case where a virtual rotation is a multiple of the
    # ciphertext size, and so the ciphertexts are simply reordered.
    ciphertext_size = 4
    num_ciphertexts = 2
    actual = vos_vos_erkin(num_ciphertexts, ciphertext_size, SWAP_TWO_CIPHERTEXTS)
    assert len(actual) == 1
    run_network_implementation(num_ciphertexts, ciphertext_size, SWAP_TWO_CIPHERTEXTS)


def test_reorder_three_ciphertexts():
    # this tests the case where a virtual rotation is a multiple of the
    # ciphertext size, and so the ciphertexts are simply reordered.
    ciphertext_size = 4
    num_ciphertexts = 3
    actual = vos_vos_erkin(num_ciphertexts, ciphertext_size, REORDER_THREE_CIPHERTEXTS)
    assert len(actual) == 1
    run_network_implementation(num_ciphertexts, ciphertext_size, REORDER_THREE_CIPHERTEXTS)
