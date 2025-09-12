"""An implementation of the graph coloring approach of Vos-Vos-Erkin 2022 from
http://dx.doi.org/10.1007/978-3-031-17140-6_20
"""

import itertools
from dataclasses import dataclass
import networkx as nx
from typing import Iterable


@dataclass(frozen=True)
class RotationGroup:
    """A group of indices to rotate."""

    indices: frozenset[int]


@dataclass(frozen=True)
class SourceShift:
    source: int
    shift: int


@dataclass(frozen=True)
class SourceShiftBits:
    source: int
    shift: int
    shift_bits: list[int]


def is_power_of_two(n: int) -> bool:
    """Check if n is a power of two."""
    return n & (n - 1) == 0


def vos_vos_erkin(n: int, mapping: Iterable[tuple[int, int]]) -> list[RotationGroup]:
    assert is_power_of_two(n)

    sources = {source for (source, _) in mapping}
    format_string = f"{{:0{n.bit_length() - 1}b}}"

    # LSB-to-MSB ordering of bits of each shift
    source_shift_bits: list[SourceShiftBits] = []
    for (source, target) in mapping:
        shift = (target - source) % n
        source_shift_bits.append(
            SourceShiftBits(
                source=source,
                shift=shift,
                shift_bits=[int(b) for b in reversed(format_string.format(shift))],
            )
        )

    # Here we compute the coresponding table of values after each rotation,
    # akin to the table in Figure 3 of the paper, excluding the first column
    # of values that are about to be rotated by 1.
    rounds: list[dict[SourceShift, int]] = []
    rounds.append(
        {
            SourceShift(source=ssb.source, shift=ssb.shift): ssb.source
            for ssb in source_shift_bits
        }
    )
    print()
    print(rounds[-1])
    for i in range(n.bit_length() - 1):
        rotation_amount = 1 << i
        last_round = rounds[-1]
        current_round = {}
        for ssb in source_shift_bits:
            key = SourceShift(source=ssb.source, shift=ssb.shift)
            next_position = last_round[key]
            if ssb.shift_bits[i] == 1:
                next_position = (last_round[key] + rotation_amount) % n
            current_round[key] = next_position
        rounds.append(current_round)
        print(rounds[-1])
    print()

    # Any two sources with colliding values in a round require an edge in G.
    G = nx.Graph()
    for round_num, round in enumerate(rounds):
        for (ss1, ss2) in itertools.combinations(round.keys(), 2):
            if ss1.source != ss2.source and round[ss1] == round[ss2]:
                print(
                    f"Round {round_num}: collision between {ss1} and {ss2} at {round[ss1]}"
                )
                G.add_edge(ss1.source, ss2.source)

    # Vertices are added as edges are added, so no vertices implies no edges,
    # and all sources can be rotated together.
    if G.number_of_nodes() == 0:
        return [RotationGroup(indices=frozenset(sources))]
    coloring = nx.coloring.greedy_color(G, strategy="saturation_largest_first")

    indices_by_color = [[] for _ in range(1 + max(coloring.values()))]
    for index, color in coloring.items():
        indices_by_color[color].append(index)

    return [RotationGroup(indices=frozenset(group)) for group in indices_by_color]
