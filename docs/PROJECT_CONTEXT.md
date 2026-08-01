# Project context

This repository turns a virtual-pet/VLA discussion into a standalone, reproducible project. The original idea was a simple simulated creature with an egocentric camera and internal drives: it should find and eat food when satiety is low, interact with toys when bored, and react to a physical environment.

The project deliberately separates two concerns that are often mixed in demos. The **Gym** is the trusted source of physics, tasks, reward, observations, seeded replay, and training data. The **toy room** is a user-facing view of one Gym episode; it can show drives, accept an approved task request, and replay a trajectory, but it must not own hidden state or hand-script action outcomes.

The initial build uses Gymnasium and MuJoCo. It starts with parameterized skills rather than raw joint torques so it can validate closed-loop perception, drive prioritization, failure recovery, data collection, and evaluation before adding a harder articulated-control track. Fire Boy/Tiny Toybox remains a useful reference for an inspectable bounded action contract, but is not a dependency.

## Canonical decisions

- Use `satiety` rather than an ambiguously decreasing hunger number: `1.0` means full.
- Keep the v1 action set small: `walk_to`, `pick_up`, `consume`, `place`, `run_around`, and `idle`.
- Make all action outcomes explicit; invalid actions never silently mutate state.
- Support `state_oracle`, `hybrid`, and egocentric `rgb` observations, and report their results separately.
- Write one replayable, versioned record per step. Do not mix privileged simulator state into RGB-only training data.
- Establish a scripted oracle and state-based RL baseline before investing in VLM/VLA integration or a polished frontend.
