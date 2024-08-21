from vos_vos_erkin import vos_vos_erkin, RotationGroup


def test_fig3():
    n = 16
    permutation = {
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
    }
    actual = vos_vos_erkin(n, permutation)
    assert len(actual) == 2
    for bad_edge in [
        (1, 2),
        (1, 3),
        (4, 5),
        (8, 9),
        (11, 12),
        (12, 13),
        (14, 15),
    ]:
        assert not any(
            bad_edge[0] in group.indices and bad_edge[1] in group.indices
            for group in actual
        )
