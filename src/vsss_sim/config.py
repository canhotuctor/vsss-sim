"""
IEEE Very Small Size Soccer (VSSS) specification constants.

References
----------
- IEEE VSSS 2023 ruleset: https://small-size.informatik.uni-bremen.de/rules
- Field: 150 cm × 130 cm inner dimensions
- Teams: 3 robots per side (3 v 3)
- Ball: orange golf ball (42.67 mm diameter)
"""

from __future__ import annotations

from enum import Enum


class InitMode(str, Enum):
    """Robot and ball placement strategy used at episode/kickoff reset.

    KICKOFF : standard formation with small jitter (default).
    RANDOM  : uniformly random positions in each team's half; random headings.
              Future: SELECTOR — a learned/heuristic model outputs placements.
    """
    KICKOFF = "kickoff"
    RANDOM = "random"

# ---------------------------------------------------------------------------
# Field (all dimensions in metres)
# ---------------------------------------------------------------------------
FIELD_LENGTH: float = 1.50   # x-axis extent (150 cm)
FIELD_WIDTH: float = 1.30    # y-axis extent (130 cm)

# Goal
GOAL_WIDTH: float = 0.40     # 40 cm opening
GOAL_DEPTH: float = 0.10     # 10 cm deep behind end-line

# Corner chamfers (cut at 45° with this leg length on each edge)
FIELD_CHAMFER: float = 0.07  # 7 cm

# Center circle radius (used for kickoff clearance)
CENTER_RADIUS: float = 0.10  # 10 cm

# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------
N_TEAMS: int = 2
N_ROBOTS: int = 3            # per team (3 v 3 format)

TEAM_BLUE: int = 0           # blue team index
TEAM_YELLOW: int = 1         # yellow team index

# ---------------------------------------------------------------------------
# Ball
# ---------------------------------------------------------------------------
BALL_RADIUS: float = 0.02135  # golf ball: 42.67 mm diameter → 21.35 mm radius
BALL_MASS: float = 0.046      # kg  (~45.9 g golf ball)
BALL_FRICTION: float = 0.05   # rolling-friction coefficient (μ_r)
BALL_WALL_RESTITUTION: float = 0.80
BALL_ROBOT_RESTITUTION: float = 0.80

# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------
ROBOT_SIZE: float = 0.075                # max 7.5 cm × 7.5 cm footprint
ROBOT_RADIUS: float = ROBOT_SIZE / 2.0   # collision radius (≈ bounding circle)
ROBOT_MASS: float = 0.312                 # kg
ROBOT_INERTIA: float = 8.4375e-05        # kg·m² (solid square: m*(L²+W²)/12)
ROBOT_WALL_RESTITUTION: float = 0.20

# Differential-drive geometry
ROBOT_WHEELBASE: float = 0.053           # wheel-to-wheel distance (m)
ROBOT_WHEEL_RADIUS: float = 0.026        # wheel radius (m)
ROBOT_MAX_MOTOR_SPEED: float = 50.0      # rad/s → ~1.30 m/s linear
ROBOT_MAX_WHEEL_SPEED: float = ROBOT_MAX_MOTOR_SPEED * ROBOT_WHEEL_RADIUS  # ≈ 1.30 m/s

# Maximum wheel-speed change rate (m/s²). Wheel commands are slewed toward the
# target each physics sub-step at this rate, modelling motor torque limits and
# producing smoother (less jerky) trajectories. 1 g ≈ 9.81 m/s² is a typical
# bound for VSSS-class robots; 0 → max takes ~133 ms (~8 control steps).
ROBOT_WHEEL_ACCEL_LIMIT: float = 1.0 * 9.81  # m/s² (≈ 1 g)

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
FPS: float = 15.0           # control loop frequency in-game (Hz)   
DT: float = 1.0 / FPS       # control loop timestep in-game (s)
SUB_STEPS: int = 4          # number of physics sub-steps per control loop step

MAX_EPISODE_STEPS: int = 300  # 20 s at 15 Hz (config.FPS control steps)
KICKOFF_CLEAR_DIST: float = 0.20  # minimum robot distance from centre at kickoff

# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------
# Small dense reward per step proportional to the ball's forward displacement
# (Δx for the blue team; blue attacks +x). Telescopes to coef * (x_end - x_start)
# over an episode, so per-episode shaping is bounded by coef * FIELD_LENGTH.
BALL_FORWARD_REWARD_COEF: float = 0.10

# ---------------------------------------------------------------------------
# Normalisation headroom factor for velocity observations (ensures values
# slightly above max wheel speed are still within a reasonable range)
VELOCITY_NORM_HEADROOM: float = 1.5
# ---------------------------------------------------------------------------
RENDER_SCALE: float = 350.0   # pixels per metre
RENDER_MARGIN: int = 60       # pixel border around the field

# Colours (RGB)
COLOR_FIELD = (0, 150, 0)
COLOR_FIELD_LINES = (255, 255, 255)
COLOR_GOAL_BLUE = (50, 100, 220)
COLOR_GOAL_YELLOW = (220, 180, 0)
COLOR_ROBOT_BLUE = (30, 100, 230)
COLOR_ROBOT_YELLOW = (230, 190, 0)
COLOR_ROBOT_OUTLINE = (0, 0, 0)
COLOR_BALL = (255, 128, 0)
COLOR_BALL_OUTLINE = (200, 80, 0)
COLOR_BACKGROUND = (40, 40, 40)

# Minimum ball rendering radius in pixels (ensures ball is always visible)
MIN_BALL_RENDER_RADIUS: int = 3
