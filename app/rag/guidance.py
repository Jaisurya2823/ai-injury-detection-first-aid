"""
GuidanceEngine
==============
Generates rich first-aid guidance for a detected injury:
  - explanation     : what the injury is, what causes it
  - medicines       : list of {name, purpose, dose_note, warning}
  - steps           : numbered handling instructions
  - when_to_seek    : specific red-flag criteria for this injury
  - evidence        : RAG-retrieved source documents
"""
from __future__ import annotations
from .retriever import LocalRAG
from app.safety.rules import SafetyDecision


# ── Per-injury knowledge ───────────────────────────────────────────────────────

_INJURY_KNOWLEDGE: dict[str, dict] = {

    "cut": {
        "explanation": (
            "A cut (laceration) is a break in the skin caused by a sharp object. "
            "Depending on depth it may injure underlying tissue, nerves, or blood vessels. "
            "Proper cleaning and closure are essential to prevent infection and scarring."
        ),
        "medicines": [
            {
                "name": "Povidone-Iodine (Betadine)",
                "purpose": "Antiseptic — kills bacteria around the wound",
                "dose_note": "Apply diluted (1:10 with water) around the wound edges once",
                "warning": "Do not pour directly into deep wounds; can delay healing",
            },
            {
                "name": "Bacitracin / Neosporin ointment",
                "purpose": "Topical antibiotic — prevents infection and keeps wound moist",
                "dose_note": "Apply a thin layer after cleaning; cover with a sterile bandage",
                "warning": "Some people are allergic to neomycin in Neosporin; switch to bacitracin if rash occurs",
            },
            {
                "name": "Ibuprofen (Advil / Motrin) 200–400 mg",
                "purpose": "Pain relief and mild anti-inflammatory",
                "dose_note": "Take every 4–6 hours with food; max 1200 mg/day OTC",
                "warning": "Avoid if allergic to NSAIDs, or with stomach ulcers",
            },
            {
                "name": "Acetaminophen (Tylenol) 500–1000 mg",
                "purpose": "Pain relief (alternative to ibuprofen)",
                "dose_note": "Take every 4–6 hours; max 3000 mg/day",
                "warning": "Do not exceed dose; harmful to liver in overdose",
            },
        ],
        "steps": [
            "Wash your hands thoroughly with soap and water before touching the wound.",
            "Apply firm direct pressure with a clean cloth or sterile gauze for 10–15 minutes without lifting — this stops bleeding.",
            "Once bleeding stops, rinse the wound under clean running water for 5 minutes to remove debris.",
            "Gently clean around (not inside) the wound with mild soap, then pat dry.",
            "Apply a thin layer of antibiotic ointment (Bacitracin or Neosporin).",
            "Cover with a sterile adhesive bandage or gauze pad secured with medical tape.",
            "Change the dressing daily and keep the wound clean and moist.",
            "Watch for infection signs: increasing redness, warmth, swelling, pus, or fever.",
        ],
        "when_to_seek": [
            "Bleeding does not stop after 15 minutes of firm pressure",
            "The cut is deeper than 0.5 cm or has jagged edges that won't stay closed",
            "The cut is on the face, hands, feet, or genitals",
            "You can see fat, muscle, or bone in the wound",
            "The wound is from a rusty object and your tetanus vaccine is not up to date",
            "Signs of infection appear: spreading redness, pus, fever above 38°C / 100.4°F",
        ],
    },

    "abrasion": {
        "explanation": (
            "An abrasion (scrape or graze) is a superficial wound where the top layers of skin "
            "are removed by friction against a rough surface. It is usually painful because nerve "
            "endings are exposed. Thorough cleaning is the most important step."
        ),
        "medicines": [
            {
                "name": "Chlorhexidine solution (Hibiclens)",
                "purpose": "Antiseptic wash — effective against bacteria and fungi",
                "dose_note": "Dilute and gently rinse the abrasion once after initial water rinse",
                "warning": "Keep out of eyes and ears; rinse off after use",
            },
            {
                "name": "Bacitracin ointment",
                "purpose": "Topical antibiotic — prevents infection",
                "dose_note": "Apply a thin layer and cover with a non-stick dressing",
                "warning": "Discontinue if rash or allergic reaction occurs",
            },
            {
                "name": "Ibuprofen (Advil) 200–400 mg",
                "purpose": "Pain relief",
                "dose_note": "Take every 4–6 hours with food as needed",
                "warning": "Avoid with stomach ulcers or NSAID allergy",
            },
        ],
        "steps": [
            "Rinse the abrasion under cool running water for at least 5 minutes to flush out dirt and debris.",
            "If debris remains, gently remove with clean tweezers sterilized with alcohol.",
            "Clean around the wound with mild soap; avoid harsh scrubbing inside the wound.",
            "Apply antiseptic solution (diluted chlorhexidine or povidone-iodine) around the edges.",
            "Apply antibiotic ointment (Bacitracin) to keep the wound moist and prevent infection.",
            "Cover with a non-stick sterile dressing or bandage.",
            "Change dressing daily or when it gets wet or dirty.",
            "Monitor for infection signs: increasing redness, swelling, warmth, or pus.",
        ],
        "when_to_seek": [
            "Abrasion covers a large area (larger than your palm)",
            "Deep gravel or debris cannot be removed at home",
            "Signs of infection appear within 24–48 hours",
            "The wound is on the face and concerns you cosmetically",
            "The person has diabetes or a weakened immune system",
        ],
    },

    "bruise": {
        "explanation": (
            "A bruise (contusion) occurs when blunt trauma ruptures small blood vessels under "
            "the skin, causing blood to pool in the tissue. The area turns black/blue, then "
            "yellow-green as the body reabsorbs blood over 1–2 weeks. "
            "Most bruises heal without treatment but pain and swelling can be managed."
        ),
        "medicines": [
            {
                "name": "Ibuprofen (Advil / Motrin) 200–400 mg",
                "purpose": "Reduces pain and inflammation",
                "dose_note": "Take every 4–6 hours with food; max 1200 mg/day OTC",
                "warning": "May slightly increase bruising by thinning blood; take only as needed",
            },
            {
                "name": "Acetaminophen (Tylenol) 500–1000 mg",
                "purpose": "Pain relief without blood-thinning effect (preferred for bruises)",
                "dose_note": "Take every 4–6 hours; max 3000 mg/day",
                "warning": "Do not exceed dose",
            },
            {
                "name": "Topical Arnica gel / cream",
                "purpose": "Natural anti-inflammatory — reduces bruise discolouration",
                "dose_note": "Apply to unbroken skin 2–3 times daily",
                "warning": "Do not apply on broken skin or open wounds",
            },
        ],
        "steps": [
            "Apply a cold compress or ice pack wrapped in a cloth to the bruise immediately — do this for 20 minutes.",
            "Repeat cold therapy every 1–2 hours for the first 24–48 hours.",
            "Elevate the injured area above heart level to reduce swelling.",
            "Rest the affected area; avoid activities that increase pain.",
            "After 48 hours, switch to a warm compress to help reabsorb blood.",
            "Take acetaminophen or ibuprofen for pain as needed.",
            "Apply topical arnica gel to intact skin to speed healing.",
            "Monitor: if bruising spreads rapidly, becomes very large, or is very painful, seek evaluation.",
        ],
        "when_to_seek": [
            "Severe swelling, deformity, or inability to bear weight (may indicate fracture)",
            "Bruise is extremely painful or the skin is tight over the bruise",
            "Bruising appears without a cause or bruises appear frequently",
            "The bruise is over a joint and movement is severely limited",
            "The person is on blood thinners (warfarin, aspirin, etc.)",
        ],
    },

    "burn": {
        "explanation": (
            "Burns are tissue injuries caused by heat, chemicals, electricity, or radiation. "
            "Superficial (1st degree) burns affect only the outer skin layer causing redness. "
            "Partial-thickness (2nd degree) burns cause blistering. "
            "Full-thickness (3rd degree) burns destroy all skin layers and require immediate emergency care."
        ),
        "medicines": [
            {
                "name": "Ibuprofen (Advil / Motrin) 200–400 mg",
                "purpose": "Pain relief and anti-inflammatory (preferred for burns)",
                "dose_note": "Take every 4–6 hours with food",
                "warning": "Avoid with stomach ulcers or NSAID allergy",
            },
            {
                "name": "Acetaminophen (Tylenol) 500–1000 mg",
                "purpose": "Pain relief (alternative to ibuprofen)",
                "dose_note": "Take every 4–6 hours; max 3000 mg/day",
                "warning": "Do not exceed dose",
            },
            {
                "name": "Silver sulfadiazine cream (SSD 1%)",
                "purpose": "Prescription antibiotic cream for partial-thickness burns",
                "dose_note": "Applied by healthcare provider; not for home use without guidance",
                "warning": "Avoid if allergic to sulfa drugs; requires prescription",
            },
            {
                "name": "Aloe vera gel (pure)",
                "purpose": "Soothing and moisturising for minor superficial burns",
                "dose_note": "Apply a thin layer to cooled burn; do not use on blisters",
                "warning": "Use only pure aloe vera without alcohol or additives",
            },
        ],
        "steps": [
            "Stop the burning process: remove from heat source, extinguish flames (stop-drop-roll), move away from chemicals.",
            "Cool the burn immediately under cool (not cold/iced) running water for at least 10–20 minutes.",
            "Remove jewellery, watches, or tight clothing near the burn BEFORE swelling starts.",
            "Do NOT use ice, iced water, butter, toothpaste, or any household remedies — these cause further damage.",
            "Do NOT burst blisters — they protect against infection.",
            "Cover the burn loosely with cling film or a clean non-fluffy bandage. Do not wrap tightly.",
            "Take ibuprofen or acetaminophen for pain relief.",
            "For chemical burns: brush off dry chemical first, then rinse with large amounts of water for 20+ minutes.",
            "For electrical burns: do NOT touch the person until the power source is confirmed off.",
            "Seek emergency care for any burn larger than the person's palm, on face/hands/feet/genitals, or if blisters form.",
        ],
        "when_to_seek": [
            "Any burn larger than 3 inches (7.5 cm) in diameter",
            "Burns on face, hands, feet, genitals, major joints, or that encircle a limb",
            "Burns that appear white, brown, or black (3rd degree)",
            "Chemical or electrical burns — always require evaluation",
            "Smoke inhalation is suspected",
            "The person is under 5 years old, over 60, or has diabetes or a heart condition",
        ],
    },

    "swelling": {
        "explanation": (
            "Swelling (oedema) after an injury is caused by fluid accumulation in tissue as part of "
            "the inflammatory response. It often accompanies sprains, strains, contusions, or fractures. "
            "The PRICE method (Protect, Rest, Ice, Compress, Elevate) is the standard first-aid approach."
        ),
        "medicines": [
            {
                "name": "Ibuprofen (Advil / Motrin) 200–400 mg",
                "purpose": "Reduces swelling and inflammation (NSAID)",
                "dose_note": "Take every 4–6 hours with food; max 1200 mg/day OTC",
                "warning": "Avoid with stomach ulcers, kidney disease, or NSAID allergy",
            },
            {
                "name": "Naproxen (Aleve) 220 mg",
                "purpose": "Longer-acting anti-inflammatory (1 tablet lasts 8–12 hours)",
                "dose_note": "Take 1–2 tablets every 8–12 hours; max 660 mg/day OTC",
                "warning": "Same cautions as ibuprofen; do not combine with other NSAIDs",
            },
            {
                "name": "Acetaminophen (Tylenol) 500–1000 mg",
                "purpose": "Pain relief (does not reduce swelling)",
                "dose_note": "Take every 4–6 hours; max 3000 mg/day",
                "warning": "Do not exceed dose",
            },
        ],
        "steps": [
            "Protect: stop the activity that caused the injury and protect the area from further harm.",
            "Rest: avoid putting weight or stress on the injured area for 24–48 hours.",
            "Ice: apply an ice pack wrapped in a cloth for 20 minutes every 2–3 hours for the first 48–72 hours.",
            "Compress: wrap the area with an elastic compression bandage (not too tight — check circulation).",
            "Elevate: raise the injured limb above heart level to reduce swelling.",
            "Take ibuprofen or naproxen to reduce swelling and pain.",
            "After 48–72 hours switch from ice to gentle heat to promote healing.",
            "Gently begin range-of-motion exercises as pain allows after 48 hours.",
        ],
        "when_to_seek": [
            "Deformity visible — may indicate a fracture or dislocation",
            "Cannot bear weight on an injured ankle, knee, or foot",
            "Numbness, tingling, or loss of circulation in the area",
            "Swelling does not improve after 72 hours with PRICE treatment",
            "Joint feels unstable or 'gives way'",
        ],
    },

    "other": {
        "explanation": (
            "The system detected an injury that does not fit one of the primary categories. "
            "General first-aid principles apply: ensure scene safety, protect the injured person, "
            "control any bleeding, and seek professional evaluation."
        ),
        "medicines": [
            {
                "name": "Ibuprofen (Advil) 200–400 mg",
                "purpose": "General pain relief and anti-inflammatory",
                "dose_note": "Every 4–6 hours with food as needed",
                "warning": "Avoid with NSAID allergy or stomach ulcers",
            },
            {
                "name": "Acetaminophen (Tylenol) 500–1000 mg",
                "purpose": "General pain relief",
                "dose_note": "Every 4–6 hours; max 3000 mg/day",
                "warning": "Do not exceed dose",
            },
        ],
        "steps": [
            "Ensure the scene is safe for both you and the injured person.",
            "Have the person sit or lie down in a comfortable position.",
            "Assess the injury: look for bleeding, deformity, swelling, or skin changes.",
            "Control any bleeding with direct pressure using a clean cloth.",
            "Keep the person warm and calm; do not give food or drink if severe injury is possible.",
            "Seek professional evaluation to get an accurate diagnosis and treatment plan.",
        ],
        "when_to_seek": [
            "Any doubt about the severity of the injury",
            "Loss of consciousness, confusion, or altered mental state",
            "Severe pain, deformity, or inability to use the injured area",
        ],
    },

    "unknown": {
        "explanation": (
            "The system could not identify the injury type with sufficient confidence. "
            "This may be due to image quality, lighting, or an injury outside the system's training scope. "
            "Please capture a clearer image or consult a medical professional."
        ),
        "medicines": [],
        "steps": [
            "Capture a clearer image with good lighting and the injury fully visible.",
            "If the injury is actively bleeding, apply firm direct pressure with a clean cloth.",
            "Keep the person still and comfortable.",
            "Seek professional medical evaluation for an accurate diagnosis.",
        ],
        "when_to_seek": [
            "The injury is causing significant pain or distress",
            "There is active bleeding that is difficult to control",
            "Any concern about the severity of the injury",
        ],
    },
}


# ── Emergency overrides ────────────────────────────────────────────────────────

_EMERGENCY_STEPS = [
    "CALL EMERGENCY SERVICES (911 / 112 / your local number) IMMEDIATELY.",
    "Do not leave the person alone.",
    "Keep the person still, warm, and calm while help arrives.",
    "Control severe bleeding with firm direct pressure; do not remove embedded objects.",
    "Do not give food or drink.",
    "Be ready to perform CPR if the person becomes unresponsive and stops breathing normally.",
]

_EMERGENCY_MEDICINES: list[dict] = []   # No OTC medicines for emergencies — call for help


# ── GuidanceEngine ─────────────────────────────────────────────────────────────

class GuidanceEngine:
    def __init__(self, rag: LocalRAG):
        self.rag = rag

    def generate(self, labels: list[str], decision: SafetyDecision, context: dict) -> dict:
        risk = decision.risk_level

        # ── Emergency / Urgent path ───────────────────────────────────────────
        if risk in {"emergency_or_urgent", "urgent"}:
            primary = labels[0] if labels else "unknown"
            kb = _INJURY_KNOWLEDGE.get(primary, _INJURY_KNOWLEDGE["other"])
            evidence = self.rag.retrieve(
                " ".join(labels) + " emergency bleeding burn wound", top_k=3
            )
            return {
                "explanation": kb["explanation"],
                "medicines": _EMERGENCY_MEDICINES,
                "steps": _EMERGENCY_STEPS,
                "when_to_seek": [],
                "warning": decision.warning,
                "risk_level": risk,
                "evidence": [e.__dict__ for e in evidence],
            }

        # ── Insufficient evidence path ────────────────────────────────────────
        if risk == "insufficient_evidence" or not labels or labels == ["unknown"]:
            evidence = self.rag.retrieve("wound injury first aid", top_k=2)
            return {
                "explanation": _INJURY_KNOWLEDGE["unknown"]["explanation"],
                "medicines": [],
                "steps": _INJURY_KNOWLEDGE["unknown"]["steps"],
                "when_to_seek": _INJURY_KNOWLEDGE["unknown"]["when_to_seek"],
                "warning": decision.warning,
                "risk_level": risk,
                "evidence": [e.__dict__ for e in evidence],
            }

        # ── Normal path: merge knowledge for all detected labels ──────────────
        primary = labels[0]
        kb = _INJURY_KNOWLEDGE.get(primary, _INJURY_KNOWLEDGE["other"])

        # Merge additional medicines from secondary labels (no duplicates)
        all_medicines = list(kb["medicines"])
        seen_names = {m["name"] for m in all_medicines}
        for extra_label in labels[1:]:
            extra_kb = _INJURY_KNOWLEDGE.get(extra_label, {})
            for med in extra_kb.get("medicines", []):
                if med["name"] not in seen_names:
                    all_medicines.append(med)
                    seen_names.add(med["name"])

        # Combine when_to_seek from all labels
        all_when = list(kb.get("when_to_seek", []))
        for extra_label in labels[1:]:
            extra_kb = _INJURY_KNOWLEDGE.get(extra_label, {})
            for item in extra_kb.get("when_to_seek", []):
                if item not in all_when:
                    all_when.append(item)

        # RAG evidence
        query = " ".join(labels) + " " + risk + " first aid wound treatment medicine"
        evidence = self.rag.retrieve(query, top_k=3)

        return {
            "explanation": kb["explanation"],
            "medicines": all_medicines,
            "steps": kb["steps"],
            "when_to_seek": all_when,
            "warning": decision.warning,
            "risk_level": risk,
            "evidence": [e.__dict__ for e in evidence],
        }
