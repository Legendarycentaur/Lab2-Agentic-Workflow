import json
from typing import Any, Dict, List, Optional, Tuple


def build_state_prompt(
    agent_query: str,
    trajectory: List[Dict[str, Any]],
    step: int,
    max_steps:int,
) -> str:
    return (
        f"Agents Query: {agent_query}\n,"
        f"Step: {step}/{max_steps}\n,"
        f"Trajectory so far (JSON): {json.dumps(trajectory)}\n,"
    )