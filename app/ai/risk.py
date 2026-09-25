from __future__ import annotations

class RiskSeverityEngine:
    """Preliminary visual/context risk routing. Not a clinical assessment."""

    def estimate(self, labels: list[str], confidence: float, extent: float, context: dict) -> str:
        if confidence <= 0 or not labels or "unknown" in labels:
            return "insufficient_evidence"
        red_flags = set(context.get("red_flags", []))
        if red_flags:
            return "severe"
        bleeding = context.get("bleeding_status")
        pain = context.get("pain_level")
        movement = context.get("movement_limitation")
        if bleeding in {"persistent/heavy", "heavy"} or pain == "severe" or movement == "major":
            return "severe"
        if any(x in labels for x in ("burn", "cut", "abrasion")) and (confidence >= 0.85 or extent >= 0.20):
            return "moderate"
        return "mild"
