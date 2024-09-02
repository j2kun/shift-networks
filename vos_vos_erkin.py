"""An implementation of the graph coloring approach of Vos-Vos-Erkin 2022 from
http://dx.doi.org/10.1007/978-3-031-17140-6_20
"""

import itertools
from dataclasses import dataclass
import networkx as nx


@dataclass(frozen=True)
class RotationGroup:
    """A group of indices to rotate."""

    indices: frozenset[int]


def is_power_of_two(n: int) -> bool:
    """Check if n is a power of two."""
    return n & (n - 1) == 0


def vos_vos_erkin(n: int, permutation: dict[int, int]) -> list[RotationGroup]:
    assert is_power_of_two(n)
    assert set(permutation.keys()) == set(range(n))

    shifts = [(permutation[i] - i) % n for i in range(n)]
    format_string = f"{{:0{n.bit_length() - 1}b}}"

    # LSB-to-MSB ordering of bits of each shift
    shift_bits = [
        [int(b) for b in reversed(format_string.format(shift))] for shift in shifts
    ]

    # Here we compute the coresponding table of values after each rotation,
    # akin to the table in Figure 3 of the paper, excluding the first column
    # of values that are about to be rotated by 1.
    rounds = []
    for i in range(n.bit_length() - 1):
        rotation_amount = 1 << i
        last_round = rounds[-1] if rounds else {x: x for x in range(n)}
        rounds.append(
            {
                x: (last_round[x] + rotation_amount if bits[i] == 1 else x)
                for (x, bits) in zip(range(n), shift_bits)
            }
        )
        # print(rounds[-1])

    # Any two keys with colliding values in a round require an edge in G.
    G = nx.Graph()
    for round in rounds:
        for x, y in itertools.combinations(round.keys(), 2):
            if round[x] == round[y]:
                G.add_edge(x, y)

    coloring = nx.coloring.greedy_color(G, strategy="saturation_largest_first")

    indices_by_color = [[] for _ in range(1 + max(coloring.values()))]
    for index, color in coloring.items():
        indices_by_color[color].append(index)

    return [
        RotationGroup(indices=frozenset(group)) for group in indices_by_color
    ]
