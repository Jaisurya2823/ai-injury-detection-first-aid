from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class SafetyDecision:
    risk_level: str
    action: str
    warning: str
    reasons: list[str]

class SafetyEngine:
    """Deterministic safety gate. It does not diagnose. It routes based on visible-system output and user-reported red flags."""
    RED_FLAG_KEYS={"heavy_bleeding","continuous_bleeding","breathing_difficulty","unconscious","confusion","major_burn","chemical_burn","electrical_burn","serious_eye_injury"}

    def decide(self, labels:list[str], confidence:float, severity:str, context:dict) -> SafetyDecision:
        red_flags=set(context.get("red_flags", [])) & self.RED_FLAG_KEYS
        if red_flags:
            return SafetyDecision("emergency_or_urgent", "Seek urgent professional/emergency medical evaluation.", "The system detected a reported red flag; do not rely on automated first-aid guidance alone.", sorted(red_flags))
        if not labels or any(label in {"unknown", "other"} for label in labels):
            return SafetyDecision("insufficient_evidence", "Capture a clearer image or seek professional evaluation.", "The system could not identify a supported injury class reliably enough for specific guidance.", ["unsupported_or_unknown_class"])
        if confidence < 0.70 or severity == "insufficient_evidence":
            return SafetyDecision("insufficient_evidence", "Capture a clearer image or seek professional evaluation.", "The AI result is not reliable enough for automated guidance.", ["low_confidence_or_insufficient_evidence"])
        if severity == "severe":
            return SafetyDecision("urgent", "Seek prompt professional medical evaluation.", "Visible findings may warrant professional assessment.", ["preliminary_severity_high"])
        return SafetyDecision("routine_first_aid", "Use basic first-aid guidance and monitor for worsening.", "This is preliminary visual screening, not a medical diagnosis.", [])
