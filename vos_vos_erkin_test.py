from computational_model import Ciphertext
from vos_vos_erkin import vos_vos_erkin, implement_shift_network


FIG3_MAPPING = {
    0: 13,
    1: 8,
    2: 4,
    3: 0,
    4: 11,
    5: 7,
    6: 14,
    7: 5,
    8: 15,
    9: 3,
    10: 12,
    11: 6,
    12: 10,
    13: 2,
    14: 9,
    15: 1,
}.items()


FULL_REPLICATION_MAPPING = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 9),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 14),
    (0, 15),
]

TWO_REPLICATION_MAPPING = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (1, 8),
    (1, 9),
    (1, 10),
    (1, 11),
    (1, 12),
    (1, 13),
    (1, 14),
    (1, 15),
]


def run_network_implementation(n, mapping, shift_order=None):
    rot_groups = vos_vos_erkin(n, mapping, shift_order=shift_order)
    input = Ciphertext(list(range(n)))
    output = implement_shift_network(
        n, input, mapping, rot_groups, shift_order=shift_order
    )
    expected = [0] * n
    for source, target in mapping:
        expected[target] = input.data[source]
    expected = Ciphertext(expected)
    assert output.data == expected.data


def test_fig3():
    n = 16
    actual = vos_vos_erkin(n, FIG3_MAPPING)
    assert len(actual) == 3
    for bad_edge in [
        (0, 14),
        (0, 15),
        (1, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 9),
        (8, 9),
        (11, 12),
        (11, 13),
        (12, 13),
        (14, 15),
    ]:
        assert not any(
            bad_edge[0] in group.indices and bad_edge[1] in group.indices
            for group in actual
        )
    run_network_implementation(n, FIG3_MAPPING)


def test_mapping():
    n = 16
    actual = vos_vos_erkin(n, FULL_REPLICATION_MAPPING)
    assert len(actual) == 1
    run_network_implementation(n, FULL_REPLICATION_MAPPING)


def test_mapping_2():
    n = 16
    actual = vos_vos_erkin(n, TWO_REPLICATION_MAPPING)
    # the default ordering of shifts creates the conflict
    assert len(actual) == 2
    run_network_implementation(n, TWO_REPLICATION_MAPPING)


def test_mapping_2_with_different_ordering():
    n = 16
    actual = vos_vos_erkin(n, TWO_REPLICATION_MAPPING, shift_order=[8, 4, 2, 1])
    # Putting 8 first allows the initial value in slot 1 to be shifted
    # away from the conflict first.
    assert len(actual) == 1
    run_network_implementation(n, TWO_REPLICATION_MAPPING, shift_order=[8, 4, 2, 1])
