"""Parametric drone CAD (build123d): parts, assembly, mass properties, exports.

This package imports nothing heavy. ``schema`` and ``artifacts`` are pure Python/pydantic; the
modules that need build123d (``parts``, ``assembly``, ``mass_props``, ``export``, ``render``) are
imported explicitly by whoever builds a model.
"""
