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
    power_of_two_shifts_needed: set[int]


def is_power_of_two(n: int) -> bool:
    """Check if n is a power of two."""
    return n & (n - 1) == 0


def vos_vos_erkin(
    n: int, mapping: Iterable[tuple[int, int]], available_shifts=None
) -> list[RotationGroup]:
    assert is_power_of_two(n)
    if not available_shifts:
        # use the default order of LSB to MSB.
        available_shifts = [1 << i for i in range(n.bit_length() - 1)]
        print(f"{available_shifts=}")

    sources = {source for (source, _) in mapping}

    source_shift_bits: list[SourceShiftBits] = []
    for source, target in mapping:
        shift = (target - source) % n
        source_shift_bits.append(
            SourceShiftBits(
                source=source,
                shift=shift,
                power_of_two_shifts_needed=set(
                    x for x in available_shifts if shift & x
                ),
            )
        )
        print(f"{source_shift_bits[-1]=}")

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
    for rotation_amount in available_shifts:
        last_round = rounds[-1]
        current_round = {}
        for ssb in source_shift_bits:
            key = SourceShift(source=ssb.source, shift=ssb.shift)
            next_position = last_round[key]
            if rotation_amount in ssb.power_of_two_shifts_needed:
                next_position = (last_round[key] + rotation_amount) % n
            current_round[key] = next_position
        rounds.append(current_round)
        print(rounds[-1])
    print()

    # Any two sources with colliding values in a round require an edge in G.
    G = nx.Graph()
    for round_num, round in enumerate(rounds):
        for ss1, ss2 in itertools.combinations(round.keys(), 2):
            if ss1.source != ss2.source and round[ss1] == round[ss2]:
                print(
                    f"Round {round_num}: collision between "
                    f"{ss1} and {ss2} at {round[ss1]}"
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
