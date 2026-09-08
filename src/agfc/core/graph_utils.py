from __future__ import annotations

def _connected_component(seed: str, adjacency: dict[str, set[str]]) -> set[str]:
    stack = [seed]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(sorted(adjacency.get(current, set()) - seen))
    return seen
