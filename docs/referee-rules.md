# Referee rule contract

The match referee follows the supplied **RoboCore Futebol Mini/VSSS, version
3.0, revised 27 May 2025** (distributed as
`RoboCore_Futebol_Mini_VSS_Regras_RCX_2026.pdf`). Simulation rule changes affect
both training and evaluation; results from older referee versions are not
directly comparable.

- **8.1, scoring:** the entire ball must cross the goal line within the mouth.
  A ball straddling the line can continue through the opening; it is not
  reflected or counted early. Physics stops its remaining substeps after the
  first goal so the adapters resolve the event once.
- **2.4 and 9.2, area geometry:** any part of a rotated robot touching the area
  counts. The whole ball must lie inside it. The area is the union of the
  70 x 15 cm rectangle and the front arc, 20 cm wide and 5 cm deep. The arc is
  modeled as a circular segment determined by that chord and depth, using 32
  chords (under 0.06 mm chord error). The renderer uses the same geometry. The old
  rectangular overlap-fraction helper remains a geometry diagnostic; its
  fraction is no longer the membership criterion.
- **9.2, attacks:** an attack needs the ball in the area and an attacking
  robot in that area within 8 cm of the ball's centre. It does not require a
  defending robot. The first attacker is retained until it leaves or the ball
  leaves; ball re-entry creates a new attack. A second area occupant is a foul
  only when it touches the ball or a player involved in that attack. Rotated
  square contact uses a 0.1 mm solver tolerance rather than a proximity radius.
- **9.2.2, defending:** a participating second defender awards a penalty
  immediately during an attack, with a goal taking precedence as advantage.
  Two idle occupants do not create a penalty deferred until the ball leaves.
  A defender entering under observed opponent contact and a directed push,
  while disputing the ball, is exempt until it leaves the area or play stops.
- **9.1 and 14, timers:** uncontested goalkeeper possession and contested
  goal-area stalemate expire at five seconds. Once possession starts, moving
  the ball inside the area or briefly losing contact does not erase the timer.
  An attacking dispute resets it; the ball leaving the area ends possession.
- **15, stopped goal-area ball:** retain the newer five-second penalty for
  an uncontested ball left at rest in the defending area, including when no
  goalkeeper has touched it. Contested goalkeeper play uses the impasse rule.
- **9.1 and 12, personal contact:** observed initiated footprint contact away
  from the ball can be called anywhere on the field. Ball-between pushes are
  permitted. Ordinary personal fouls award a free kick; goalkeeper blocking
  and the specific area/possession violations retain their prescribed restarts.
- **12, free kicks:** TL uses the same marked spot as PK. The kicker is
  behind the ball, supporting attackers are in their own half, and defenders
  touch their goal-area boundary with axis-aligned footprints.
- **15, free balls:** after five seconds stopped outside the areas, use the
  quadrant where the stoppage began. A repeated failure to move the ball puts
  it at centre without moving players. A third puts it at centre and places
  all players outside the centre circle. Actual ball movement (1 mm position
  tolerance or 1 mm/s speed) clears the recurrence chain. Referee placement
  itself is not treated as ball movement.
- **7, kickoffs:** initial possession is chosen by a seeded coin toss. After
  a goal the conceding team receives the kickoff. Defending footprints stay
  outside the centre circle. Random curriculum resets remain available, but
  in-match kickoffs use legal formations even with `init_mode="random"`.

The referee remains an approximation of human judgment in a 2D simulator:
contact intent, damaging impacts and toppling cannot be established from its
state. Contact attribution uses observed motion and final footprint contact;
it cannot recover a transient contact missed between control samples.
False-start warnings/cards, temporary robot exclusion, match periods and
tournament administration are not implemented. Legal restart formations are
fixed choices from the allowed placements, not a placement strategy.

These changes do not add drills, modify PPO terminal/bootstrap behavior,
change policy inputs, or change self-play opponent selection. Referee runtime
state has new history fields; policy snapshot tensor shapes stay compatible.
