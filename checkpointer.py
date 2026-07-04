"""Checkpoint store setup — SqliteSaver for dev, swap for PostgresSaver in prod."""
from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from config import CHECKPOINT_DB


def get_checkpointer() -> SqliteSaver:
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    return SqliteSaver(conn)
