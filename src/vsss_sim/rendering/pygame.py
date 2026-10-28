"""
Pygame-based renderer for the VSSS environment.

Supports two render modes:
- ``"human"``     – opens a live window (requires a display).
- ``"rgb_array"`` – returns a ``(H, W, 3)`` uint8 NumPy array suitable for
                    video recording without opening any window.

GPU / headless note
-------------------
When ``render_mode="rgb_array"`` is used, the surface is created with
``pygame.Surface`` (no display required) and pixel data is returned via
``pygame.surfarray.array3d``.  This path is safe on headless CI runners
and inside Docker containers.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional

import numpy as np

from .. import config

try:
    import pygame
    from pygame import gfxdraw  # anti-aliased primitives
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False
    gfxdraw = None  # type: ignore[assignment]


# After a longer stall (e.g. JIT compilation), restart the frame schedule
# instead of racing through the backlog.
_MAX_PACING_LAG = 0.25  # s


def _require_pygame() -> None:
    if not _PYGAME_AVAILABLE:
        raise ImportError(
            "pygame is required for rendering. "
            "Install it with: pip install pygame>=2.4"
        )


class VSSRenderer:
    """
    Renders the VSSS simulation state to a Pygame surface.

    Parameters
    ----------
    render_mode : ``"human"`` | ``"rgb_array"``
    scale : float
        Pixels per metre (default: ``config.RENDER_SCALE``).
    margin : int
        Border width in pixels (default: ``config.RENDER_MARGIN``).
    fps : float, optional
        Human-mode playback rate in simulated physics sub-steps per wall-clock
        second: the frame rate when each frame shows one sub-step. ``None``
        renders as fast as possible.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        scale: float = config.RENDER_SCALE,
        margin: int = config.RENDER_MARGIN,
        fps: Optional[float] = None,
        baseline_fps: float = config.FPS * config.SUB_STEPS,
        control_dt: float = config.DT,
    ) -> None:
        _require_pygame()
        self.render_mode = render_mode
        self.scale = scale
        self.margin = margin
        self._fps = fps
        self._baseline_fps = baseline_fps
        self._control_dt = control_dt

        self._field_px_w = int(config.FIELD_LENGTH * scale)
        self._field_px_h = int(config.FIELD_WIDTH * scale)
        self._win_w = self._field_px_w + 2 * margin
        self._win_h = self._field_px_h + 2 * margin

        self._surface: Optional[pygame.Surface] = None
        self._screen: Optional[pygame.Surface] = None
        self._next_frame_t: Optional[float] = None

        # Measured simulated substeps per wall-clock second, over recent frames.
        self._speedup_samples: deque[tuple[float, int]] = deque(maxlen=30)
        self._last_render_t: Optional[float] = None
        self._hud_font: Optional[pygame.font.Font] = None
        self._score_font: Optional[pygame.font.Font] = None

    # Coordinate helpers

    def _to_px(self, x: float, y: float) -> tuple[int, int]:
        """World coordinates (m) → pixel coordinates."""
        px = int((x + config.FIELD_LENGTH / 2.0) * self.scale) + self.margin
        py = int((-y + config.FIELD_WIDTH / 2.0) * self.scale) + self.margin
        return px, py

    def _m_to_px(self, metres: float) -> int:
        return max(1, int(metres * self.scale))

    # Initialisation

    def _init(self) -> None:
        if self._surface is not None:
            return
        pygame.init()
        if self.render_mode == "human":
            self._screen = pygame.display.set_mode((self._win_w, self._win_h))
            pygame.display.set_caption("VSSS Simulator")
        self._surface = pygame.Surface((self._win_w, self._win_h))

    # Drawing primitives

    def _draw_field(self, surf: pygame.Surface) -> None:
        surf.fill(config.COLOR_BACKGROUND)

        hl = config.FIELD_LENGTH / 2.0
        hw = config.FIELD_WIDTH / 2.0
        c = config.FIELD_CHAMFER
        # Chamfered field boundary, ordered clockwise from the top-left.
        corners_world = [
            (-hl + c,  hw),
            ( hl - c,  hw),
            ( hl,      hw - c),
            ( hl,     -hw + c),
            ( hl - c, -hw),
            (-hl + c, -hw),
            (-hl,     -hw + c),
            (-hl,      hw - c),
        ]
        field_poly = [self._to_px(x, y) for x, y in corners_world]
        gfxdraw.filled_polygon(surf, field_poly, config.COLOR_FIELD)
        gfxdraw.aapolygon(surf, field_poly, config.COLOR_FIELD)
        gfxdraw.aapolygon(surf, field_poly, config.COLOR_FIELD_LINES)

        mid_x = self.margin + self._field_px_w // 2
        pygame.draw.aaline(
            surf, config.COLOR_FIELD_LINES,
            (mid_x, self.margin),
            (mid_x, self.margin + self._field_px_h),
        )

        cx, cy = self._to_px(0.0, 0.0)
        r_px = self._m_to_px(config.CENTER_RADIUS)
        gfxdraw.aacircle(surf, cx, cy, r_px, config.COLOR_FIELD_LINES)

        # Free-ball marks (BL), 37.5 cm from each end and 25 cm from each side.
        cross_half_length = 0.03
        for sign_x in (-1.0, 1.0):
            for sign_y in (-1.0, 1.0):
                mark_x = sign_x * config.FREE_BALL_MARK_X
                mark_y = sign_y * config.FREE_BALL_MARK_Y
                pygame.draw.aaline(
                    surf,
                    config.COLOR_FIELD_LINES,
                    self._to_px(mark_x - cross_half_length, mark_y),
                    self._to_px(mark_x + cross_half_length, mark_y),
                )
                pygame.draw.aaline(
                    surf,
                    config.COLOR_FIELD_LINES,
                    self._to_px(mark_x, mark_y - cross_half_length),
                    self._to_px(mark_x, mark_y + cross_half_length),
                )

        goal_px_h = self._m_to_px(config.GOAL_WIDTH)
        goal_px_d = self._m_to_px(config.GOAL_DEPTH)

        gx_left, gy_top = self._to_px(
            -config.FIELD_LENGTH / 2.0 - config.GOAL_DEPTH,
            config.GOAL_WIDTH / 2.0,
        )
        pygame.draw.rect(
            surf, config.COLOR_GOAL_BLUE,
            pygame.Rect(gx_left, gy_top, goal_px_d, goal_px_h),
        )
        pygame.draw.rect(
            surf, config.COLOR_FIELD_LINES,
            pygame.Rect(gx_left, gy_top, goal_px_d, goal_px_h),
            2,
        )

        gx_right, gy_top = self._to_px(
            config.FIELD_LENGTH / 2.0,
            config.GOAL_WIDTH / 2.0,
        )
        pygame.draw.rect(
            surf, config.COLOR_GOAL_YELLOW,
            pygame.Rect(gx_right, gy_top, goal_px_d, goal_px_h),
        )
        pygame.draw.rect(
            surf, config.COLOR_FIELD_LINES,
            pygame.Rect(gx_right, gy_top, goal_px_d, goal_px_h),
            2,
        )

        # Goal areas — 3 inner sides each (end-line is shared with the field border).
        ga_half_y = config.GOAL_AREA_LENGTH_Y / 2.0
        ga_x = config.GOAL_AREA_LENGTH_X
        hl = config.FIELD_LENGTH / 2.0

        for side_x, sign in ((-hl, 1), (hl, -1)):
            inner_x = side_x + sign * ga_x
            for wx, wy in (
                ((inner_x, -ga_half_y), (inner_x, ga_half_y)),
                ((side_x, ga_half_y), (inner_x, ga_half_y)),
                ((side_x, -ga_half_y), (inner_x, -ga_half_y)),
            ):
                pygame.draw.aaline(
                    surf, config.COLOR_FIELD_LINES,
                    self._to_px(*wx), self._to_px(*wy),
                )
            radius = config.GOAL_AREA_ARC_RADIUS
            extent = math.asin(config.GOAL_AREA_ARC_HALF_WIDTH / radius)
            arc = [
                self._to_px(
                    inner_x + sign * (config.GOAL_AREA_ARC_DEPTH - radius + radius * math.cos(angle)),
                    radius * math.sin(angle),
                )
                for angle in np.linspace(-extent, extent, 33)
            ]
            pygame.draw.aalines(surf, config.COLOR_FIELD_LINES, False, arc)

    def _draw_robot(
        self,
        surf: pygame.Surface,
        x: float,
        y: float,
        theta: float,
        team: int,
    ) -> None:
        half = config.ROBOT_SIZE / 2.0
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Rotate the four corners of the square into world coords, then to pixels
        corners_px = [
            self._to_px(
                x + cos_t * lx - sin_t * ly,
                y + sin_t * lx + cos_t * ly,
            )
            for lx, ly in ((-half, -half), (half, -half), (half, half), (-half, half))
        ]

        color = (
            config.COLOR_ROBOT_BLUE
            if team == config.TEAM_BLUE
            else config.COLOR_ROBOT_YELLOW
        )
        # Filled body with AA edge — gfxdraw.filled_polygon has a hard edge, so
        # overlay aapolygon (1 px AA) in the same colour to smooth it, then a
        # second aapolygon in the outline colour for the visible border.
        gfxdraw.filled_polygon(surf, corners_px, color)
        gfxdraw.aapolygon(surf, corners_px, color)
        gfxdraw.aapolygon(surf, corners_px, config.COLOR_ROBOT_OUTLINE)

        cx, cy = self._to_px(x, y)
        tip = self._to_px(x + cos_t * half * 0.85, y + sin_t * half * 0.85)
        pygame.draw.aaline(surf, config.COLOR_ROBOT_OUTLINE, (cx, cy), tip)

    def _draw_ball(self, surf: pygame.Surface, x: float, y: float) -> None:
        cx, cy = self._to_px(x, y)
        r_px = max(config.MIN_BALL_RENDER_RADIUS, self._m_to_px(config.BALL_RADIUS))
        # filled_circle + aacircle in the fill colour gives an AA filled disc;
        # a second aacircle in the outline colour adds the visible border.
        gfxdraw.filled_circle(surf, cx, cy, r_px, config.COLOR_BALL)
        gfxdraw.aacircle(surf, cx, cy, r_px, config.COLOR_BALL)
        gfxdraw.aacircle(surf, cx, cy, r_px, config.COLOR_BALL_OUTLINE)

    def _draw_score(
        self,
        surf: pygame.Surface,
        score_blue: int,
        score_yellow: int,
    ) -> None:
        if self._score_font is None:
            try:
                self._score_font = pygame.font.SysFont("monospace", 22, bold=True)
            except Exception:
                return
        text = self._score_font.render(
            f"Blue {score_blue}  :  {score_yellow} Yellow",
            True,
            config.COLOR_FIELD_LINES,
        )
        surf.blit(text, (self._win_w // 2 - text.get_width() // 2, 8))

    def _pace(self, simulated_substeps: int) -> None:
        """Hold the displayed frame for the wall time of its sub-steps.

        Deadlines accumulate, so a slow frame (policy inference, display sync)
        is absorbed by shorter following frames rather than slowing playback.
        Every frame is still shown.
        """
        now = time.perf_counter()
        if self._next_frame_t is None or now - self._next_frame_t > _MAX_PACING_LAG:
            self._next_frame_t = now
        self._next_frame_t += simulated_substeps / self._fps
        delay = self._next_frame_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    def _update_speedup_sample(self, simulated_substeps: int) -> None:
        """Record elapsed wall time and simulated steps since the prior frame."""
        now = time.perf_counter()
        if self._last_render_t is not None:
            dt = now - self._last_render_t
            if dt > 0:
                self._speedup_samples.append((dt, simulated_substeps))
        self._last_render_t = now

    def _hud_font_or_none(self) -> Optional["pygame.font.Font"]:
        if self._hud_font is None:
            try:
                self._hud_font = pygame.font.SysFont("monospace", 16, bold=True)
            except Exception:
                return None
        return self._hud_font

    def _draw_speedup(self, surf: pygame.Surface) -> None:
        """Draw the achieved simulation speed relative to real time."""
        if not self._speedup_samples:
            return
        elapsed = sum(dt for dt, _ in self._speedup_samples)
        simulated_substeps = sum(steps for _, steps in self._speedup_samples)
        speedup_rate = (
            simulated_substeps / (elapsed * self._baseline_fps)
            if elapsed > 0 and self._baseline_fps > 0
            else 0.0
        )

        font = self._hud_font_or_none()
        if font is None:
            return
        text = font.render(
            f"speedup {speedup_rate:4.1f}x", True, config.COLOR_FIELD_LINES
        )
        surf.blit(text, (8, 8))

    def _draw_step(self, surf: pygame.Surface, step: int) -> None:
        """Draw the current control-step count in the bottom-left corner."""
        font = self._hud_font_or_none()
        if font is None:
            return
        sim_t = step * self._control_dt
        # Extra spaces after "step" and "episode" align the two numeric
        # columns vertically (see _draw_episode below).
        text = font.render(
            f"step  {step:>5d}  ({sim_t:5.2f}s)", True, config.COLOR_FIELD_LINES,
        )
        surf.blit(text, (8, self._win_h - font.get_linesize() - 8))

    def _draw_episode(self, surf: pygame.Surface, episode: int) -> None:
        """Draw the episode counter just above the step counter."""
        font = self._hud_font_or_none()
        if font is None:
            return
        text = font.render(
            f"episode   {episode:d}", True, config.COLOR_FIELD_LINES,
        )
        surf.blit(text, (8, self._win_h - 2 * font.get_linesize() - 8))

    # Public render method

    def render(
        self,
        ball: np.ndarray,
        robots: np.ndarray,
        score: np.ndarray,
        step: Optional[int] = None,
        episode: Optional[int] = None,
        simulated_substeps: int = 1,
    ) -> Optional[np.ndarray]:
        """
        Render one frame.

        Parameters
        ----------
        ball : (4,) [x, y, vx, vy]
        robots : (N_TEAMS, N_ROBOTS, 6)
        score : (2,) int  [blue_goals, yellow_goals]
        step : optional control-step counter; when given, drawn in the bottom
            left in human mode.
        episode : optional episode counter; when given, drawn just above the
            step counter in human mode.
        simulated_substeps : number of physics substeps represented by this
            frame, used to calculate the achieved speedup in human mode.

        Returns
        -------
        frame : ndarray (H, W, 3) uint8 if ``render_mode="rgb_array"``, else None.
        """
        self._init()
        surf = self._surface

        self._draw_field(surf)

        for team in range(config.N_TEAMS):
            for r_idx in range(config.N_ROBOTS):
                rx, ry, theta = (
                    robots[team, r_idx, 0],
                    robots[team, r_idx, 1],
                    robots[team, r_idx, 2],
                )
                self._draw_robot(surf, rx, ry, theta, team)

        self._draw_ball(surf, ball[0], ball[1])
        self._draw_score(surf, int(score[0]), int(score[1]))

        if self.render_mode == "human":
            self._update_speedup_sample(simulated_substeps)
            self._draw_speedup(surf)
            if step is not None:
                self._draw_step(surf, int(step))
            if episode is not None:
                self._draw_episode(surf, int(episode))
            self._screen.blit(surf, (0, 0))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit(0)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise SystemExit(0)
            if self._fps is not None:
                self._pace(simulated_substeps)
            return None

        raw = pygame.surfarray.array3d(surf)   # (W, H, 3)
        return np.transpose(raw, (1, 0, 2))    # → (H, W, 3)

    def close(self) -> None:
        """Destroy the renderer and quit Pygame if it was initialised."""
        if self._surface is not None:
            self._surface = None
        if self._screen is not None:
            pygame.display.quit()
            self._screen = None
        if _PYGAME_AVAILABLE:
            pygame.quit()
