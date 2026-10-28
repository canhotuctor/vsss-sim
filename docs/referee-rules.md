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
- **9.2.2, defending:** a participating second defender during an attack is
  a foul, but the rule spares the defenders if the attackers score, so play
  continues as advantage. The penalty is awarded when the attack ends without
  a goal (the whole ball is no longer in the area), after five seconds, or as
  soon as any other call would stop play; the earlier defender foul wins. A goal
  during the advantage stands and cancels the penalty. Two idle occupants are
  not a foul. A defender entering under observed opponent contact and a
  directed push, while disputing the ball, is exempt until it leaves the area
  or play stops.
- **9.1 and 14, timers:** uncontested goalkeeper possession and contested
  goal-area stalemate expire at five seconds. Once possession starts, moving
  the ball inside the area or briefly losing contact does not erase the timer.
  An attacking dispute resets it; the ball leaving the area ends possession.
- **Stuck ball:** every stoppage rule treats a ball at or below 0.1 m/s as
  stuck, matching VSSReferee's `ballMinSpeedForStuck`. A ball pinned between
  robots jitters and is never perfectly still.
- **15, stopped goal-area ball:** retain the newer five-second penalty for
  an uncontested ball stuck in the defending area, including when no
  goalkeeper has touched it. Contested goalkeeper play uses the impasse rule.
- **9.1 and 12, personal contact:** the rulebook calls only collisions that
  disrupt play or could damage a robot, and the simulator cannot see intent.
  A personal foul therefore needs all of:
  - *avoidable*: the victim spent at least 0.2 s in the offender's forward
    lane (35 cm ahead, one robot width wide) before contact, which is the time
    to stop from full speed at 1 g plus one decision. A robot cutting across a
    driving opponent cannot draw a foul;
  - *one-sided*: the offender's wheel drive toward the victim exceeds the
    victim's drive toward the offender by 0.3 m/s, so mutual crashes play on;
  - *no ball dispute*: both robots within 10 cm of the ball is legal contact,
    replacing the old ball-between check;
  - and either a fresh impact (no contact one step earlier) at 0.6 m/s or
    more on a victim within 20 cm of the ball, a fresh impact at 0.9 m/s or
    more anywhere, or a push or pin sustained for one second.

  Drive speeds come from wheel speeds: robots are kinematic and a collision
  splits the overlap equally, so displacement cannot show who pushed.
  Personal fouls award a free kick; goalkeeper blocking and the specific
  area/possession violations retain their prescribed restarts.
- **9.2.1 and 14, goalkeeper block:** contact with the goalkeeper is ordinary
  play. Only a personal foul, as defined above, by an attacker inside the area
  on the goalkeeper is a block, restarted with a goal kick for the goalkeeper's
  team. Pushing the goalkeeper while both dispute the ball stays legal.
- **Team symmetry:** awarded restart positions and robot IDs rotate by 180
  degrees when teams swap. This includes supporting players and defenders;
  reflecting only x changes the ordered observations seen by a shared policy.
  If both teams qualify for the same referee call in one control interval,
  seeded equal-probability priority resolves the ambiguity, rather than always
  selecting array index zero. Exact midfield-axis ties for free-ball
  quadrants use seeded neutral choices, retained throughout the stoppage. Goals discarded during a stopped frame cannot
  reach adapter goal metrics or terminate a training episode.
- **12, free kicks:** TL uses the same marked spot as PK. The kicker is
  behind the ball, supporting attackers are in their own half, and defenders
  touch their goal-area boundary with axis-aligned footprints.
- **15, free balls:** after five seconds stuck outside the areas, use the
  quadrant where the stoppage began. A repeated failure to move the ball puts
  it at centre without moving players. A third puts it at centre and places
  all players outside the centre circle. Actual ball movement (1 mm position
  tolerance or 1 mm/s speed) clears the recurrence chain. Referee placement
  itself is not treated as ball movement.
- **Referee modes:** `full` (default) applies every call above. `simplified`
  keeps only the two-robot goal-area fouls (9.2.1 and 9.2.2, with advantage)
  and frees a ball stuck anywhere with a free ball, goal areas included, so
  play cannot stall there. `off` stops play only for goals.
- **7, kickoffs:** initial possession is chosen by a seeded coin toss. After
  a goal the conceding team receives the kickoff. Defending footprints stay
  outside the centre circle. Random curriculum resets remain available, but
  in-match kickoffs use legal formations even with `init_mode="random"`.

The referee remains an approximation of human judgment in a 2D simulator:
contact intent, damaging impacts and toppling cannot be established from its
state. Personal-foul attribution uses wheel drive, lane history and footprint
contact at control samples; it cannot recover a transient contact missed
between them. The pushed-in exemption still attributes contact from observed
motion.
False-start warnings/cards, temporary robot exclusion, match periods and
tournament administration are not implemented. Legal restart formations are
fixed choices from the allowed placements, not a placement strategy.

These changes do not add drills, modify PPO terminal/bootstrap behavior,
change policy inputs, or change self-play opponent selection. Referee runtime
state has new history fields; policy snapshot tensor shapes stay compatible.
