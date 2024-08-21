# I have forgotten what I intended to do with this nonsense
from ortools.sat.python import cp_model

num_ssa_values = 16
depth = 4
vector_width = 8

all_ssa_values = range(num_ssa_values)
all_levels = range(depth)
all_widths = range(vector_width)

model = cp_model.CpModel()

slot_assignments = {}
for s in all_ssa_values:
    for l in all_levels:
        for w in all_widths:
            # A slot variable assigns an ssa value S
            # at a given depth L to the index w of a vector v
            slot_assignments[(s, l, w)] = model.NewBoolVar(f"slot_s{s}_l{l}_w{w}")


rotations = {}
for l in all_levels:
    for w in all_widths:
        # A rotation variable inserts a rotation of a given shift
        # at a given level.
        rotations[(l, w)] = model.NewBoolVar(f"shift_l{l}_w{w}")

