"""Hermes directory-plugin entry. Implementation lives in the kerdoios package."""
from __future__ import annotations

from .kerdoios import EXPLAIN_SCHEMA, INVENTORY_SCHEMA, PLAN_SCHEMA, RECORD_SCHEMA, register

__all__ = ["register", "PLAN_SCHEMA", "EXPLAIN_SCHEMA", "INVENTORY_SCHEMA", "RECORD_SCHEMA"]
