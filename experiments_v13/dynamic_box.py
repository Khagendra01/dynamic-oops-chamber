"""DynamicBox re-export for the v1.3 layer.

The obstacle dynamics (piecewise-constant velocity schedules, exact wall
reflections, appearance/disappearance) are reused unchanged from the v1.2.1
``experiments_v12.dynamic_chamber`` implementation.  Import path alias so
v1.3 code depends on a stable local name without modifying v1.2 sources.
"""
from __future__ import annotations

from experiments_v12.dynamic_chamber import DynamicBox, VelocityEvent, to_dynamic

__all__ = ["DynamicBox", "VelocityEvent", "to_dynamic"]
