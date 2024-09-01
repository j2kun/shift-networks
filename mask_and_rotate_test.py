from computational_model import Ciphertext
from mask_and_rotate import mask_and_rotate


def verify(input, output, permutation):
    for i, x in enumerate(input.data):
        assert output.data[permutation[i]] == x


def test_vos_vos_erkin_example():
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
    input = Ciphertext(list(range(n)))
    actual = mask_and_rotate(input, permutation)
    verify(input, actual, permutation)
