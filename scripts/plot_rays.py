"""Draw the agent, its 7 rays and a circle / diamond at the same position."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

from pathlib import Path

from catperc import config as C
from catperc.geometry import CIRCLE, DIAMOND, DIAMOND_HALF_DIAG, ray_distances, sense

OBJ_X, OBJ_Y = 12.0, 110.0

fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
for ax, shape, name in zip(axes, [CIRCLE, DIAMOND], ["circle", "diamond"]):
    d = np.minimum(np.asarray(ray_distances(OBJ_X, OBJ_Y, shape)), C.RAY_MAX_LENGTH)
    inputs = np.asarray(sense(OBJ_X, OBJ_Y, shape))

    ax.add_patch(Circle((0, 0), C.AGENT_RADIUS, color="0.3"))
    for angle, length, inp in zip(C.RAY_ANGLES, d, inputs):
        end = length * np.array([np.sin(angle), np.cos(angle)])
        hit = inp > 0
        ax.plot([0, end[0]], [0, end[1]], color="tab:red" if hit else "0.75", lw=1.5)
        if hit:
            ax.plot(*end, "o", color="tab:red", ms=4)

    if shape == CIRCLE:
        ax.add_patch(Circle((OBJ_X, OBJ_Y), C.CIRCLE_RADIUS, fill=False, lw=2))
    else:
        h = DIAMOND_HALF_DIAG
        verts = [(OBJ_X + h, OBJ_Y), (OBJ_X, OBJ_Y + h), (OBJ_X - h, OBJ_Y), (OBJ_X, OBJ_Y - h)]
        ax.add_patch(Polygon(verts, fill=False, lw=2))

    ax.set_title(f"{name}: inputs " + " ".join(f"{v:.1f}" for v in inputs), fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-70, 70)
    ax.set_ylim(-20, 230)
    ax.set_xlabel("x (agent-centred)")
axes[0].set_ylabel("y")
fig.suptitle("Ray sensing: same position, different shapes")
fig.tight_layout()


out = Path(__file__).resolve().parent.parent / "figures" / "rays.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"saved {out}")