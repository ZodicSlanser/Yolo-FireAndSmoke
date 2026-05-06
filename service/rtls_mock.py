RTLS_PRESENCE: dict[str, list[dict]] = {
    "production_line_3": [
        {"id": "EMP-1142", "name": "Khalid M.", "role": "Line Supervisor"},
        {"id": "EMP-2034", "name": "Ahmed S.", "role": "Operator"},
    ],
    "warehouse_a": [
        {"id": "EMP-3201", "name": "Yusuf K.", "role": "Forklift Operator"},
    ],
    "loading_bay_2": [],
}


def lookup(zone: str) -> dict:
    candidates = RTLS_PRESENCE.get(zone, [])
    return {
        "method": "rtls_zone_match",
        "candidates": candidates,
        "primary": candidates[0] if candidates else None,
    }
