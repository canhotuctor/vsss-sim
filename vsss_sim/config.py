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

# ---------------------------------------------------------------------------
# Field (all dimensions in metres)
# ---------------------------------------------------------------------------
FIELD_LENGTH: float = 1.50   # x-axis extent (150 cm)
FIELD_WIDTH: float = 1.30    # y-axis extent (130 cm)

# Goal
GOAL_WIDTH: float = 0.40     # 40 cm opening
GOAL_DEPTH: float = 0.10     # 10 cm deep behind end-line

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
BALL_FRICTION: float = 0.15   # rolling-friction coefficient (μ_r)
BALL_WALL_RESTITUTION: float = 0.70
BALL_ROBOT_RESTITUTION: float = 0.70

# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------
ROBOT_SIZE: float = 0.075                # max 7.5 cm × 7.5 cm footprint
ROBOT_RADIUS: float = ROBOT_SIZE / 2.0   # collision radius (≈ bounding circle)
ROBOT_MASS: float = 0.18                 # kg
ROBOT_INERTIA: float = 8.4375e-05        # kg·m² (solid square: m*(L²+W²)/12)
ROBOT_WALL_RESTITUTION: float = 0.20

# Differential-drive geometry
ROBOT_WHEELBASE: float = 0.053           # wheel-to-wheel distance (m)
ROBOT_WHEEL_RADIUS: float = 0.026        # wheel radius (m)
ROBOT_MAX_MOTOR_SPEED: float = 50.0      # rad/s → ~1.30 m/s linear
ROBOT_MAX_WHEEL_SPEED: float = ROBOT_MAX_MOTOR_SPEED * ROBOT_WHEEL_RADIUS  # ≈ 1.30 m/s

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
FPS: float = 60.0           # simulation frequency (Hz)
DT: float = 1.0 / FPS       # timestep (s)

MAX_EPISODE_STEPS: int = 1200  # 20 s at 60 Hz
KICKOFF_CLEAR_DIST: float = 0.20  # minimum robot distance from centre at kickoff

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
