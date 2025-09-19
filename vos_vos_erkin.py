"""An implementation of the graph coloring approach of Vos-Vos-Erkin 2022 from
http://dx.doi.org/10.1007/978-3-031-17140-6_20
"""

import itertools
from dataclasses import dataclass
import networkx as nx
from typing import Iterable, Optional
from computational_model import Ciphertext, Slot, is_power_of_two


MappingEntry = tuple[Slot, Slot]
Mapping = list[MappingEntry]


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
    maxLog2 = n.bit_length()
    # If the (possibly virtual) ciphertext size is exactly a power of two, then
    # the formula below includes a shift by n, which is a no-op for a cyclic
    # rotation.
    if is_power_of_two(n):
        maxLog2 -= 1

    # use the default order of 1, 2, 4, ..., ceil(log2(n))
    return [1 << i for i in range(maxLog2)]


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

        assert is_power_of_two(self.ciphertext_size)
        assert set(shift_order) == set(default_shift_order(self.n))

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
                curr_ct, curr_slot = last_round_posns[key]
                curr_virtual_slot = curr_ct * self.ciphertext_size + curr_slot

                next_position = (curr_ct, curr_slot)
                if rotation_amount in ssb.power_of_two_shifts_needed:
                    curr_virtual_slot = (curr_virtual_slot + rotation_amount) % self.n
                    next_position = (
                        curr_virtual_slot // self.ciphertext_size,
                        curr_virtual_slot % self.ciphertext_size,
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


def apply_virtual_rotation(
    input: list[Ciphertext], rotation: int, rotate_masks: list[list[int]]
) -> list[Ciphertext]:
    """Apply a virtual rotation to a real list of ciphertexts.

    A virtual ciphertext is a flattening of a list of ciphertexts. When this
    materializes to a set of rotations of the real ciphertexts, we need to
    track the movement of slots between ciphertexts, and decompose the virtual
    rotation into a set of real rotations and extra masks.

    For example, we have to deal with ciphertexts which are rotated in such a
    way that they overlap two subsequent ciphertexts in the larger "virtual"
    ciphertext. E.g. if we have size 8 and two slots 3, 7 are rotated left by
    2:

     ct0: . . . x . . . y
     ct1: . . . . . . . .

    then after their rotation if the desired target for slot 7 is ct1 slot 2,
    we have the following reality

     ct0: . y . . . x . .
     ct1: . . . . . . . .

    and we need to mask the position of y to add it to ct1, while masking out x
    to keep it with ct0.
    """
    num_ciphertexts = len(input)
    ciphertext_size = len(input[0])

    # We need to identify the (possibly two) target ciphertexts for each
    # input ciphertext that was rotated. If there is only one target---i.e., if
    # the rotation was exactly the power of two matching the
    # ciphertext_size---we can update the target with the rotated ciphertexts
    # and be done. If there are two targets, we need to add additional masks at
    # the split
    if rotation % ciphertext_size == 0:
        masked = [input_ct * mask for input_ct, mask in zip(input, rotate_masks)]
        # we don't need to apply the rotation at all, just reorder
        ciphertext_shift = rotation // ciphertext_size
        return masked[-ciphertext_shift:] + masked[:-ciphertext_shift]

    min_slot = 0
    max_slot = ciphertext_size - 1
    last_slot_before_wrap = max(
        x for x in range(ciphertext_size) if x + rotation < ciphertext_size
    )

    # Nb., there is a choice here:
    #
    #  1. Mask first, then rotate together, then mask twice to separate the
    #     two targets.
    #  2. Split mask first, mask twice, then rotate twice.
    #
    # Option (1) requires one rotation, but three ct-pt muls (depth 2), and
    # option (2) requires two rotations, but only two ct-pt muls (depth 1). Not
    # sure which is better. This func implements (2).
    results = [Ciphertext([0] * ciphertext_size) for _ in range(num_ciphertexts)]
    for i in range(num_ciphertexts):
        mask = rotate_masks[i]
        # Determine the two ciphertext targets for each input ciphertext.
        # This is a worst-case target, i.e., the target if all slots are part of
        # the mask. We will correct this later.
        target1 = i + (min_slot + rotation) // ciphertext_size
        target2 = i + (max_slot + rotation) // ciphertext_size

        # Split each of the input masks into two masks, one for the pre-split
        # and one for the post-split.
        boundary_slot = last_slot_before_wrap
        mask1 = mask[: boundary_slot + 1] + [0] * (ciphertext_size - boundary_slot - 1)
        mask2 = [0] * (boundary_slot + 1) + mask[boundary_slot + 1 :]

        # Apply the split masks to the input and rotate
        if all(x == 0 for x in mask1):
            rotated1 = None
        else:
            masked1 = input[i] * mask1
            rotated1 = masked1.rotate(rotation % ciphertext_size)

        if all(x == 0 for x in mask2):
            rotated2 = None
        else:
            masked2 = input[i] * mask2
            rotated2 = masked2.rotate(rotation % ciphertext_size)

        # Add the rotated masks to their respective targets.
        if rotated1:
            results[target1 % num_ciphertexts] += rotated1
        if rotated2:
            results[target2 % num_ciphertexts] += rotated2

    return results


def implement_one_group(
    group_init: list[Ciphertext],
    source_shifts: list[SourceShift],
    rounds: list[ShiftRound],
    group: RotationGroup,
) -> list[Ciphertext]:
    num_ciphertexts = len(group_init)
    ciphertext_size = len(group_init[0])
    current = group_init

    # Run the entire shift strategy for one rotation group
    for round_num, round in enumerate(rounds):
        if round_num == 0:
            continue

        # need two masks, one to select the sources in this group that need
        # to be rotated, and one to preserve the values at fixed positions.
        rotate_positions = []
        fixed_positions = []
        for key in source_shifts:
            current_posn = rounds[round_num - 1].positions[key]
            # we have to recompute this dynamically, because the sources
            # rotated during the ShiftStrategy setup include conflicts from
            # other rotation groups.
            if key.shift & round.rotation_amount:
                rotate_positions.append(current_posn)
            else:
                fixed_positions.append(current_posn)

        fixed_masks = [[0] * ciphertext_size for _ in range(num_ciphertexts)]
        for ct, slot in fixed_positions:
            fixed_masks[ct][slot] = 1

        # skip masking if possible
        fixed_current = []
        for ct, fixed_mask in zip(current, fixed_masks):
            if all(x == 0 for x in fixed_mask):
                fixed = None
            elif all(x == 1 for x in fixed_mask):
                fixed = ct
            else:
                fixed = ct * fixed_mask
            fixed_current.append(fixed)

        rotated_current = [None] * num_ciphertexts
        if rotate_positions:
            rotate_masks = [[0] * ciphertext_size for _ in range(num_ciphertexts)]
            for ct, slot in rotate_positions:
                rotate_masks[ct][slot] = 1
            rotated_current = apply_virtual_rotation(
                current, round.rotation_amount, rotate_masks
            )

        for i, (fixed, rotated) in enumerate(zip(fixed_current, rotated_current)):
            if not fixed and not rotated:
                continue  # current[i] is unchanged

            if not fixed:
                current[i] = rotated
            elif not rotated:
                current[i] = fixed
            else:
                current[i] = fixed + rotated

    return current


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
        group_results[group_num] = implement_one_group(
            group_results[group_num],
            source_shifts,
            rounds,
            group,
        )

    # add all the results together
    final_result = [Ciphertext([0] * len(x)) for x in input]
    for result in group_results:
        for i, ct in enumerate(result):
            if ct:
                final_result[i] += ct
    return final_result
