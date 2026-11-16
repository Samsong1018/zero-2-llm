import cube, json, os, sys
import numpy as np
import tkinter as tk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import nn
from tensor import no_grad


# Loading the model

model = cube.build_model()
data = json.load(open(os.path.join(HERE, "weights.json")))

for p, w in zip(model.parameters(), data["params"]):
    p.data = np.array(w, dtype=np.float32)

bn_layers = [l for l in model.layers if isinstance(l, nn.BatchNorm1d)]
for layer, (mean, var) in zip(bn_layers, data["bn"]):
    layer.running_mean = np.array(mean)
    layer.running_var = np.array(var)

for l in model.layers:
    l.training = False


# NN solver
def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

def beam_solve(s, K=50, max_depth=20):
    with no_grad():
        beam = [(s, 0.0, [])]
        for depth in range(max_depth):
            for st, sc, path in beam:
                if np.array_equal(st, cube.SOLVED):
                    return path

            S = np.stack([b[0] for b in beam])
            logp = log_softmax(model(S).data)
            logp = logp[:, :9]

            cands = {}
            for b, (st, sc, path) in enumerate(beam):
                for m in range(9):
                    if path and m == cube.INV[path[-1]]:
                        continue
                    child = st[cube.MOVES[m]]
                    score = sc + logp[b, m]
                    k = child.tobytes()
                    if k not in cands or score > cands[k][0]:
                        cands[k] = (score, child, path + [m])

            best = sorted(cands.values(), key=lambda c: c[0], reverse=True)[:K]
            beam = [(st, sc, path) for sc, st, path in best]
        return None


# GUI const

CELL = 50
COLORS = ["White", "Blue", "Red", "Yellow", "Green", "Orange"]
FACE_POS = [(2, 0), (4, 2), (2, 2), (2, 4), (0, 2), (6, 2)]
FIXED = (14, 18, 23)   # yellow/green/orange corner, can't be painted
paint_color = 0
palette_buttons = []


# Drawing/Cubstate

def draw(state):
    canvas.delete("all")
    for sticker in range(24):
        face = sticker // 4
        k = sticker % 4
        col = FACE_POS[face][0] + k % 2
        row = FACE_POS[face][1] + k // 2
        x0, y0 = (col*CELL, row*CELL)
        x1, y1 = ((col+1)*CELL, (row+1)*CELL)
        canvas.create_rectangle(x0, y0, x1, y1, fill=COLORS[state[sticker]], outline="black", width=2)
        if sticker in FIXED:
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            canvas.create_oval(cx-5, cy-5, cx+5, cy+5, fill="black")

def apply_mov(m):
    global state
    state = state[cube.MOVES[m]]
    draw(state)

def scramble():
    global state, cur_path
    state = cube.SOLVED.copy()
    cur_path = None
    status.config(text="")
    for z in range(25):
        m = np.random.randint(0, 9)
        state = state[cube.MOVES[m]]
    draw(state)

def reset():
    global state, cur_path
    cur_path = None
    status.config(text="")
    state = cube.SOLVED.copy()
    draw(state)


# Solve/Playback

def solve():
    global cur_path, state
    if (np.bincount(state, minlength=6) != 4).any():
        status.config(text="Error invalid input")
        return

    cur_path = beam_solve(state, K=200)
    if cur_path is None:
        status.config(text="No solution found :(")
        return

    label_text = " ".join(cube.MOVE_NAMES[m] for m in cur_path)
    status.config(text=label_text)

def play(i=0):
    global cur_path
    if cur_path is None:
        status.config(text="Press solve first")
        return
    play_but.config(state="disabled")

    if i == len(cur_path):
        cur_path = None
        play_but.config(state="normal")
        return
    apply_mov(cur_path[i])

    root.after(400, play, i+1)


# Let a user input their own scramble

def pick(c):
    status.config(text="Hold your cube with the yellow/green/orange corner at bottom-back-left (yellow down, green left, orange back")
    global paint_color
    paint_color = c
    for i, b in enumerate(palette_buttons):
        if i == c:
            b.config(relief="sunken", bd=4)
        else:
            b.config(relief="raised", bd=2)

def on_click(event):
    global state, cur_path
    sticker = None
    col = event.x // CELL
    row = event.y // CELL
    for face, (fx, fy) in enumerate(FACE_POS):
        dc = col - fx
        dr = row - fy
        if dc in (0, 1) and dr in (0, 1):
            sticker = face*4 + dr*2 + dc
            break
    if sticker is None: return
    if sticker in FIXED:
        return

    state = state.copy()
    state[sticker] = paint_color
    draw(state)
    cur_path = None


# setup GUI

root = tk.Tk()
root.title("2b2 Solver")

canvas = tk.Canvas(root, width=8*CELL, height=6*CELL)
canvas.pack()
canvas.bind("<Button-1>", on_click)

buttons = tk.Frame(root)
buttons.pack()
for m, name in enumerate(cube.MOVE_NAMES):
    button = tk.Button(buttons, text=name, width=4, command=lambda m=m: apply_mov(m))
    button.pack(side="left")

controls = tk.Frame(root)
controls.pack()
scram = tk.Button(controls, text="Scramble", command=scramble)
res = tk.Button(controls, text="Reset", command=reset)
solve_but = tk.Button(controls, text="Solve", command=solve)
play_but = tk.Button(controls, text="Play Solution", command=play)
scram.pack(side="left")
res.pack(side="left")
solve_but.pack(side="left")
play_but.pack(side="left")

status = tk.Label(root, text="", wraplength=8*CELL)
status.pack()

pallete = tk.Frame(root)
pallete.pack()
for c, col in enumerate(COLORS):
    but = tk.Button(pallete, bg=col, width=3, command=lambda c=c: pick(c))
    but.pack(side="left")
    palette_buttons.append(but)
palette_buttons[paint_color].config(relief="sunken", bd=4)


# Init

state = cube.SOLVED.copy()
cur_path = None
draw(state)

root.mainloop()
