"""Frozen, reproducible milestone harnesses.

The modules here are intentionally isolated from the authoritative simulator.
They retain the exact policy, protocol, and reporting code required to replay
historical results, while ``ecosystem_gym`` itself remains the evolving Gym,
task, replay, and viewer implementation.
"""
