"""An implementation of the graph coloring approach of Vos-Vos-Erkin 2022 from
http://dx.doi.org/10.1007/978-3-031-17140-6_20
"""

import itertools
from dataclasses import dataclass
import networkx as nx
from typing import Iterable, Optional
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


@dataclass(frozen=True)
class ShiftRound:
    # current positions of the input (source, shift) pairs in this round
    positions: dict[SourceShift, int]
    # The set of indices rotated left in this round
    rotated_indices: set[int]
    # The amount rotated left in this round
    rotation_amount: int


def default_shift_order(n: int):
    # use the default order of 1, 2, 4, ..., log2(n)
    return [1 << i for i in range(n.bit_length() - 1)]


class ShiftStrategy:
    def __init__(
        self, n: int, shift_order: Optional[list[int]] = None, debug: bool = False
    ):
        if not shift_order:
            shift_order = default_shift_order(n)

        assert is_power_of_two(n)
        assert set(shift_order) == set(1 << i for i in range(n.bit_length() - 1))

        self.n = n
        self.shift_order = shift_order
        self.debug = debug

    def evaluate(self, mapping: Iterable[tuple[int, int]]) -> list[ShiftRound]:
        source_shift_bits: list[SourceShiftBits] = []
        for source, target in mapping:
            shift = (target - source) % self.n
            needed_shifts = set(x for x in self.shift_order if shift & x)
            source_shift_bits.append(
                SourceShiftBits(
                    source=source,
                    shift=shift,
                    power_of_two_shifts_needed=needed_shifts,
                )
            )
            if self.debug:
                print(f"{source_shift_bits[-1]=}")

        # Here we compute the coresponding table of values after each rotation,
        # akin to the table in Figure 3 of the paper, including the first column
        # of values that are about to be rotated by 1.
        rounds: list[dict[SourceShift, int]] = []
        rounds.append(
            ShiftRound(
                positions={
                    SourceShift(source=ssb.source, shift=ssb.shift): ssb.source
                    for ssb in source_shift_bits
                },
                rotated_indices=set(),
                rotation_amount=0,
            )
        )
        for rotation_amount in self.shift_order:
            last_round_posns = rounds[-1].positions
            current_round_posns = {}
            current_round_rotated_indices = set()

            for ssb in source_shift_bits:
                key = SourceShift(source=ssb.source, shift=ssb.shift)
                next_position = last_round_posns[key]
                if rotation_amount in ssb.power_of_two_shifts_needed:
                    next_position = (last_round_posns[key] + rotation_amount) % self.n
                current_round_posns[key] = next_position
                current_round_rotated_indices.add(next_position)

            rounds.append(
                ShiftRound(
                    positions=current_round_posns,
                    rotated_indices=current_round_rotated_indices,
                    rotation_amount=rotation_amount,
                )
            )

        return rounds


def vos_vos_erkin(
    n: int,
    mapping: Iterable[tuple[int, int]],
    shift_order: Optional[list[int]] = None,
    debug: bool = False,
) -> list[RotationGroup]:
    strategy = ShiftStrategy(n=n, shift_order=shift_order, debug=debug)
    rounds = strategy.evaluate(mapping)
    sources = {source for (source, _) in mapping}

    # Any two sources with colliding values in a round require an edge in G.
    G = nx.Graph()
    for round_num, round in enumerate(rounds):
        if round_num == 0:
            continue  # skip the initial round which is the starting position

        posns = round.positions
        for ss1, ss2 in itertools.combinations(posns.keys(), 2):
            if ss1.source != ss2.source and posns[ss1] == posns[ss2]:
                if debug:
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
    strategy = ShiftStrategy(n=n, shift_order=shift_order)
    rounds = strategy.evaluate(mapping)

    # each rotation_group corresponds to one independent set of rotations
    group_results = [Ciphertext([0] * len(input)) for _ in rotation_groups]

    for group_num, group in enumerate(rotation_groups):
        if len(group) == 0:
            continue

        # Run the entire shift strategy for one rotation group
        current = input
        for round_num, round in enumerate(rounds):
            if round_num == 0:
                continue

            if len(round.rotated_indices) == 0:
                continue

            mask = Ciphertext(
                [
                    1 if i in round.rotated_indices and i in group.indices else 0
                    for i in range(n)
                ]
            )
            current = current * mask
            current = current.rotate(round.rotation_amount)
            group_results[group_num] += current

    # add all the results together
    final_result = Ciphertext([0] * len(input))
    for result in group_results:
        final_result += result
    return final_result
