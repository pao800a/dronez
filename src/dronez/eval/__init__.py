"""Evaluation: scenarios, metrics, closed-loop runner, report and video.

Kept import-light on purpose: ``dronez.eval.video`` runs in a child process that must set
``MUJOCO_GL`` before ``mujoco`` is imported, so this package imports nothing heavy.
"""
