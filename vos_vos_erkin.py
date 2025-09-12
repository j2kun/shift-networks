"""An implementation of the graph coloring approach of Vos-Vos-Erkin 2022 from
http://dx.doi.org/10.1007/978-3-031-17140-6_20
"""

import itertools
from dataclasses import dataclass
import networkx as nx
from typing import Iterable
from computational_model import Ciphertext, is_power_of_two


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


def default_shift_order(n: int):
    # use the default order of 1, 2, 4, ..., log2(n)
    return [1 << i for i in range(n.bit_length() - 1)]


def vos_vos_erkin(
    n: int, mapping: Iterable[tuple[int, int]], shift_order: list[int] = None
) -> list[RotationGroup]:
    assert is_power_of_two(n)
    if not shift_order:
        shift_order = default_shift_order(n)
        print(f"{shift_order=}")

    assert set(shift_order) == set(1 << i for i in range(n.bit_length() - 1))

    sources = {source for (source, _) in mapping}

    source_shift_bits: list[SourceShiftBits] = []
    for source, target in mapping:
        shift = (target - source) % n
        needed_shifts = set(x for x in shift_order if shift & x)
        source_shift_bits.append(
            SourceShiftBits(
                source=source,
                shift=shift,
                power_of_two_shifts_needed=needed_shifts,
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
    for rotation_amount in shift_order:
        last_round = rounds[-1]
        current_round = {}
        for ssb in source_shift_bits:
            key = SourceShift(source=ssb.source, shift=ssb.shift)
            next_position = last_round[key]
            if rotation_amount in ssb.power_of_two_shifts_needed:
                next_position = (last_round[key] + rotation_amount) % n
            current_round[key] = next_position
        rounds.append(current_round)

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


def implement_shift_network(
    n: int,
    input: Ciphertext,
    mapping: Iterable[tuple[int, int]],
    rotation_groups: list[RotationGroup],
    shift_order: list[int] = None,
):
    if not shift_order:
        shift_order = default_shift_order(n)

    # FIXME: undupe from above using NetworkStrategy and Rounds classes
    source_shift_bits: list[SourceShiftBits] = []
    for source, target in mapping:
        shift = (target - source) % n
        needed_shifts = set(x for x in shift_order if shift & x)
        source_shift_bits.append(
            SourceShiftBits(
                source=source,
                shift=shift,
                power_of_two_shifts_needed=needed_shifts,
            )
        )

    for accum, group in zip(group_results, rotation_groups):
        rounds: list[dict[SourceShift, int]] = []
        rounds.append(
            {
                SourceShift(source=ssb.source, shift=ssb.shift): ssb.source
                for ssb in source_shift_bits
            }
        )
        for rotation_amount in shift_order:
            last_round = rounds[-1]
            current_round = {}
            for ssb in source_shift_bits:
                key = SourceShift(source=ssb.source, shift=ssb.shift)
                next_position = last_round[key]
                if rotation_amount in ssb.power_of_two_shifts_needed:
                    next_position = (last_round[key] + rotation_amount) % n
                current_round[key] = next_position
            rounds.append(current_round)

    # each rotation_group corresponds to one ciphertext
    group_results = [Ciphertext([0] * len(input)) for _ in rotation_groups]

    final_result = Ciphertext([0] * len(input))
    for result in group_results:
        final_result += result
    return final_result
