"""
Experimental constants for the replication of Beer (2003),
"The dynamics of active categorical perception in an evolved model agent",
Adaptive Behavior 11(4): 209-243.

Tags:
  [p.X]        = stated in Beer (2003), page X
  [B96] VERIFY = Beer (2003) defers to Beer (1996); value is a placeholder until checked there
  [ours]       = our convention, not from either paper
"""
import math

# ---------------------------------------------------------------- world
OBJECT_SPEED = 3.0              # [p.214] vertical velocity -3, straight down
OFFSET_RANGE = (-50.0, 50.0)    # [p.214] initial horizontal offset
OBJECT_START_Y = 275.0          # [B96] VERIFY  must be > 200 (Fig. 5, p.220)
WORLD_WIDTH = 400.0             # [B96] VERIFY

# ---------------------------------------------------------------- agent body
AGENT_DIAMETER = 30.0           # [p.212]
AGENT_RADIUS = AGENT_DIAMETER / 2
MOTOR_GAIN = 5.0                # [B96] VERIFY  vx = MOTOR_GAIN * (o_13 - o_14)? check Eq. 1 sign/form

# ---------------------------------------------------------------- objects
CIRCLE_RADIUS = 15.0            # [p.214] R_C(phi) = 15
DIAMOND_SIDE = 30.0             # [p.214] R_D(phi) = 15*sqrt(2)*sec(phi), read Eq. on PDF
CIRCLE_ALPHA, DIAMOND_ALPHA = 0.0, 1.0   # [p.214] hybrid object parameter

# ---------------------------------------------------------------- sensors
N_RAYS = 7                      # [p.212]
RAY_SPREAD = math.pi / 6        # [p.212]
RAY_MAX_LENGTH = 220.0          # [B96] VERIFY
MAX_SENSOR_INPUT = 10.0         # [B96] VERIFY  input at ray length 0; 0 at max length [p.214]

RAY_ANGLES = tuple(             # [ours] evenly spaced, symmetric about vertical
    -RAY_SPREAD / 2 + i * RAY_SPREAD / (N_RAYS - 1) for i in range(N_RAYS)
)

# ---------------------------------------------------------------- network
N_SENSORY = 7                   # [p.214] neurons 1-7
N_INTER = 5                     # [p.214] neurons 8-12; mirror pairs 8/12, 9/11; 10 central [p.225]
N_MOTOR = 2                     # [p.214] neurons 13-14
INTER_SELF_CONNECTIONS = True   # [p.231] interneuron 10 "inhibits itself"
BILATERAL_SYMMETRY = True       # [p.214] weights, biases, time constants
SENSORY_SHARED_PARAMS = True    # [p.214] one gain, bias, tau for all sensory neurons
NON_SENSORY_GAIN = 1.0          # [p.214]

WEIGHT_RANGE = (-5.0, 5.0)      # [B96] VERIFY
BIAS_RANGE = (-5.0, 5.0)        # [B96] VERIFY
TAU_RANGE = (1.0, 2.0)          # [B96] VERIFY
SENSORY_GAIN_RANGE = (1.0, 5.0) # [B96] VERIFY

# ---------------------------------------------------------------- integration
DT = 0.1                        # [B96] VERIFY
TRIAL_DURATION = OBJECT_START_Y / OBJECT_SPEED   # [ours] until vertical separation = 0
N_STEPS = int(round(TRIAL_DURATION / DT))

# ---------------------------------------------------------------- fitness
N_EVAL_TRIALS = 24              # [p.214] uniformly spread over OFFSET_RANGE (shapes split between them?)
MAX_DISTANCE = 45.0             # [B96] VERIFY  |d| clip/normalization so p = 1 - |d| lies in [0,1]

# ---------------------------------------------------------------- evolution
ALGORITHM = "population hill climber (Back 1996)"  # [p.214]; we may use CMA-ES -> list as deviation

# ---------------------------------------------------------------- targets to reproduce
TARGET_EVAL_PERF = 0.9983       # [p.215] best agent, 24 trials, 80 generations
TARGET_RANDOM_PERF = 0.9708     # [p.215] 10,000 random trials


if __name__ == "__main__":
    print(f"Steps per trial: {N_STEPS} (duration {TRIAL_DURATION:.1f})")
    print("Ray angles (deg):", [round(math.degrees(a), 1) for a in RAY_ANGLES])