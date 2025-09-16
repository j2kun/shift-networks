"""An implementation of the graph coloring approach of Vos-Vos-Erkin 2022 from
http://dx.doi.org/10.1007/978-3-031-17140-6_20
"""

import itertools
from dataclasses import dataclass
import networkx as nx
from typing import Iterable, Optional
from computational_model import Ciphertext, is_power_of_two


Slot = tuple[int, int]
MappingEntry = tuple[Slot, Slot]


@dataclass(frozen=True)
class RotationGroup:
    """A group of source slots to rotate."""

    sources: frozenset[Slot]

    def __len__(self) -> int:
        return len(self.sources)


@dataclass(frozen=True)
class SourceShift:
    source: Slot
    shift: int


@dataclass(frozen=True)
class SourceShiftBits:
    source: Slot
    shift: int
    power_of_two_shifts_needed: set[int]


@dataclass(frozen=True)
class ShiftRound:
    # current positions of the input (source, shift) pairs in this round,
    # AFTER the shift by rotation_amount occurs
    positions: dict[SourceShift, Slot]
    # The amount rotated right in this round; for the first round this is zero
    rotation_amount: int


def default_shift_order(n: int):
    # use the default order of 1, 2, 4, ..., log2(n)
    return [1 << i for i in range(n.bit_length() - 1)]


class ShiftStrategy:
    def __init__(
        self,
        num_ciphertexts: int,
        ciphertext_size: int,
        shift_order: Optional[list[int]] = None,
        debug: bool = False,
    ):
        self.ciphertext_size = ciphertext_size
        self.num_ciphertexts = num_ciphertexts

        # Multi-ciphertext support is handled by flattening multiple
        # ciphertexts into one long, virtual ciphertext and rotating within
        # that. This introduces some suboptimality, but is a good starting
        # point.
        self.n = ciphertext_size * num_ciphertexts

        if not shift_order:
            shift_order = default_shift_order(self.n)

        assert is_power_of_two(ciphertext_size)
        assert set(shift_order) == set(
            1 << i for i in range(ciphertext_size.bit_length() - 1)
        )

        self.shift_order = shift_order
        self.debug = debug

    def virtual_shift(self, source: Slot, target: Slot) -> int:
        ct_source, slot_source = source
        ct_target, slot_target = target
        virtual_source = ct_source * self.ciphertext_size + slot_source
        virtual_target = ct_target * self.ciphertext_size + slot_target
        return (virtual_target - virtual_source) % self.n

    def evaluate(self, mapping: Iterable[MappingEntry]) -> list[ShiftRound]:
        source_shift_bits: list[SourceShiftBits] = []
        for source, target in mapping:
            shift = self.virtual_shift(source, target)
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
        rounds: list[dict[SourceShift, Slot]] = []
        rounds.append(
            ShiftRound(
                positions={
                    SourceShift(source=ssb.source, shift=ssb.shift): ssb.source
                    for ssb in source_shift_bits
                },
                rotation_amount=0,
            )
        )

        round_num = 0
        if self.debug:
            print(f"Round {round_num}: {rounds[-1]}")

        for rotation_amount in self.shift_order:
            round_num += 1
            last_round_posns = rounds[-1].positions
            current_round_posns = {}

            for ssb in source_shift_bits:
                key = SourceShift(source=ssb.source, shift=ssb.shift)
                next_position: Slot = last_round_posns[key]
                if rotation_amount in ssb.power_of_two_shifts_needed:
                    next_position_ct = next_position[0] + (
                        rotation_amount // self.ciphertext_size
                    )
                    next_position_slot = next_position[1] + (
                        rotation_amount % self.ciphertext_size
                    )
                    next_position = (
                        next_position_ct % self.num_ciphertexts,
                        next_position_slot % self.ciphertext_size,
                    )
                current_round_posns[key] = next_position

            rounds.append(
                ShiftRound(
                    positions=current_round_posns,
                    rotation_amount=rotation_amount,
                )
            )
            if self.debug:
                print(f"Round {round_num}: {rounds[-1]}")

        return rounds


def vos_vos_erkin(
    num_ciphertexts: int,
    ciphertext_size: int,
    mapping: Iterable[tuple[int, int]],
    shift_order: Optional[list[int]] = None,
    debug: bool = False,
) -> list[RotationGroup]:
    strategy = ShiftStrategy(
        num_ciphertexts=num_ciphertexts,
        ciphertext_size=ciphertext_size,
        shift_order=shift_order,
        debug=debug,
    )
    rounds = strategy.evaluate(mapping)
    sources: set[Slot] = {source for (source, _) in mapping}

    # Any two sources with colliding values in a round require an edge in G.
    G = nx.Graph()
    for source in sources:
        G.add_node(source)

    for round_num, round in enumerate(rounds):
        if round_num == 0:
            continue  # skip the initial round which is the starting position

        posns = round.positions
        for ss1, ss2 in itertools.combinations(posns.keys(), 2):
            if ss1.source != ss2.source and posns[ss1] == posns[ss2]:
                if debug:
                    print(
                        f"Round {round_num}: collision between "
                        f"{ss1} and {ss2} at {round.positions[ss1]}"
                    )
                G.add_edge(ss1.source, ss2.source)

    # Vertices are added as edges are added, so no vertices implies no edges,
    # and all sources can be rotated together.
    if G.number_of_nodes() == 0:
        return [RotationGroup(sources=frozenset(sources))]
    coloring = nx.coloring.greedy_color(G, strategy="saturation_largest_first")

    sources_by_color = [[] for _ in range(1 + max(coloring.values()))]
    for source, color in coloring.items():
        sources_by_color[color].append(source)

    return [RotationGroup(sources=frozenset(group)) for group in sources_by_color]


def implement_shift_network(
    input: list[Ciphertext],
    mapping: Iterable[tuple[int, int]],
    rotation_groups: list[RotationGroup],
    shift_order: list[int] = None,
) -> Ciphertext:
    num_ciphertexts = len(input)
    ciphertext_size = len(input[0])
    strategy = ShiftStrategy(
        num_ciphertexts=num_ciphertexts,
        ciphertext_size=ciphertext_size,
        shift_order=shift_order,
    )
    rounds = strategy.evaluate(mapping)

    # each rotation_group corresponds to one independent set of rotations
    # each input ciphertext is cloned here for simplicity
    group_results = [[x for x in input] for _ in rotation_groups]

    for group_num, group in enumerate(rotation_groups):
        if len(group) == 0:
            continue

        source_shifts = [
            SourceShift(source=source, shift=strategy.virtual_shift(source, target))
            for (source, target) in mapping
            if source in group.sources
        ]
        # Run the entire shift strategy for one rotation group
        for round_num, round in enumerate(rounds):
            if round_num == 0:
                continue

            # need two masks, one to select the sources in this group that need
            # to be rotated, and one to preserve the values at fixed sources.
            rotate_sources = []
            fixed_sources = []
            for key in source_shifts:
                current_posn = rounds[round_num - 1].positions[key]
                # we have to recompute this dynamically, because the sources
                # rotated during the ShiftStrategy setup include conflicts from
                # other rotation groups.
                if key.shift & round.rotation_amount:
                    rotate_sources.append(current_posn)
                else:
                    fixed_sources.append(current_posn)

            rotate_masks = [[0] * ciphertext_size for _ in range(num_ciphertexts)]
            for (ct, slot) in rotate_sources:
                rotate_masks[ct][slot] = 1

            fixed_masks = [[0] * ciphertext_size for _ in range(num_ciphertexts)]
            for (ct, slot) in fixed_sources:
                fixed_masks[ct][slot] = 1

            current = group_results[group_num]

            # skip masking if possible
            fixed_current = []
            for fixed_mask in fixed_masks:
                if all(x == 0 for x in fixed_mask):
                    fixed = None
                elif all(x == 1 for x in fixed_mask):
                    fixed = current
                else:
                    fixed = current * fixed_mask
                fixed_current.append(fixed)

            # skip masking if possible
            rotated_current = []
            for rotate_mask in rotate_masks:
                if all(x == 0 for x in rotate_mask):
                    rotated = None
                elif all(x == 1 for x in rotate_mask):
                    rotated = current.rotate(round.rotation_amount)
                else:
                    rotated = (current * rotate_mask).rotate(round.rotation_amount)
                rotated_current.append(rotated)

            # now we have to deal with ciphertexts which are rotated
            # in such a way that they overlap two subsequent ciphertexts
            # in the larger "virtual" ciphertext. E.g. if we have size 8
            # and two slots 3, 7 are rotated left by 2:
            #
            #  ct0: . . . x . . . y
            #  ct1: . . . . . . . .
            #
            # then after their rotation if the desired target for slot 7
            # is ct1 slot 2, we have the following reality
            #
            #  ct0: . y . . . x . .
            #  ct1: . . . . . . . .
            #
            # and we need to mask the position of y to add it to ct1,
            # while masking out x to keep it with ct0.

            # FIXME: implement this part

            # old code:
            # if not fixed:
            #     group_results[group_num] = rotated
            # elif not rotated:
            #     group_results[group_num] = fixed
            # else:
            #     group_results[group_num] = fixed + rotated

    # add all the results together
    final_result = Ciphertext([0] * len(input))
    for result in group_results:
        final_result += result
    return final_result
