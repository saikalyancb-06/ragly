"""Generate a demo corpus of field-maintenance manuals in ./samples/manuals.

Realistic enough to show auto-tuning: abbreviations with expansions, error codes,
torque specs, numbered procedures and safety warnings.
"""
from pathlib import Path

import pymupdf as fitz

OUT = Path(__file__).resolve().parent.parent / "samples" / "manuals"
OUT.mkdir(parents=True, exist_ok=True)

FAULTS = [
    ("E-47", "Low oil pressure in the tap changer", "Stop the transformer. Check oil level and top up with IEC 60296 oil."),
    ("E-52", "Over-temperature trip on the main breaker", "Wait for cool down below 60 °C, then reset the thermal relay."),
    ("E-63", "SF6 gas pressure low", "Isolate the bay and call the gas handling team. Do not operate the breaker."),
    ("F-12", "Auxiliary supply failure", "Check the 230 V auxiliary MCB and the battery charger output (110 V DC)."),
    ("F-21", "Motor drive timeout in the Ring Main Unit (RMU)", "Inspect the motor limit switch and the 40 Nm drive coupling."),
]

PROCEDURES = [
    ("ROUTINE INSPECTION OF THE RING MAIN UNIT", [
        "Isolate the Ring Main Unit (RMU) and apply the lockout tag.",
        "Verify absence of voltage with the approved voltage detector.",
        "Clean the bushings with a lint free cloth.",
        "Check the SF6 gas pressure gauge reads between 1.2 bar and 1.5 bar.",
        "Tighten the earth connection bolt to 40 Nm.",
        "Record the operation counter reading in the log book.",
    ], "WARNING: Never open the RMU cover while the busbar is live. Confirm isolation with the substation in charge."),
    ("REPLACING THE MAIN CIRCUIT BREAKER (CB) CONTACTS", [
        "De-energise the panel and earth both sides.",
        "Remove the arc chute assembly, part number P/N 88123-A.",
        "Replace the fixed and moving contacts as a matched set.",
        "Apply contact grease sparingly to the silver plated surfaces.",
        "Torque the contact stem nut to 25 Nm.",
        "Measure contact resistance; it must be below 60 micro-ohm.",
    ], "CAUTION: Contacts may retain heat. Allow 30 minutes cooling before handling."),
    ("SIX MONTHLY MAINTENANCE OF THE AUXILIARY TRANSFORMER", [
        "Switch off the Miniature Circuit Breaker (MCB) feeding the auxiliary transformer.",
        "Measure insulation resistance at 1 kV; the value must exceed 100 megaohm.",
        "Check the cooling fan current draw, rated 1.8 A.",
        "Verify the oil level sight glass is between the minimum and maximum marks.",
        "Torque the HV terminal bolts to 55 Nm.",
    ], "WARNING: Insulation testing injects high voltage. Clear all personnel before starting."),
]

SPECS = [
    ("Rated voltage", "11 kV"), ("Rated current", "630 A"), ("Short circuit rating", "20 kA for 3 s"),
    ("SF6 filling pressure", "1.4 bar at 20 °C"), ("Earth bolt torque", "40 Nm"),
    ("HV terminal bolt torque", "55 Nm"), ("Contact stem nut torque", "25 Nm"),
    ("Auxiliary supply", "230 V AC / 110 V DC"), ("Operating temperature", "-5 °C to 55 °C"),
]


def page(doc, title, body, size=10.5):
    p = doc.new_page()
    p.insert_textbox(fitz.Rect(50, 50, 545, 60 + 14), title, fontsize=13, fontname="hebo")
    p.insert_textbox(fitz.Rect(50, 85, 545, 800), body, fontsize=size)


def main():
    # 1. Fault code manual
    d = fitz.open()
    page(d, "SUBSTATION EQUIPMENT - FAULT CODE MANUAL (Rev 4)",
         "This manual lists the fault codes shown on the Ring Main Unit (RMU) and the Circuit Breaker (CB) "
         "control panel. Abbreviations: RMU (Ring Main Unit), CB (Circuit Breaker), MCB (Miniature Circuit "
         "Breaker), HV (High Voltage), LV (Low Voltage), SF6 gas insulated switchgear.\n\n"
         "Always follow the site Standard Operating Procedure (SOP) before any intervention.")
    for code, meaning, action in FAULTS:
        page(d, f"FAULT {code}",
             f"Code: {code}\nMeaning: {meaning}\n\nRecommended action:\n{action}\n\n"
             f"WARNING: Do not reset {code} more than twice without an inspection by a qualified engineer.")
    d.save(OUT / "fault_code_manual.pdf")

    # 2. Procedures
    d = fitz.open()
    page(d, "MAINTENANCE PROCEDURES - 11 kV SWITCHGEAR",
         "Scope: routine and corrective maintenance for the Ring Main Unit (RMU) and the Circuit Breaker (CB).")
    for title, steps, warning in PROCEDURES:
        body = warning + "\n\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        page(d, title, body)
    d.save(OUT / "maintenance_procedures.pdf")

    # 3. Technical data (scanned: image only, forces OCR)
    src = fitz.open()
    p = src.new_page()
    p.insert_textbox(fitz.Rect(50, 50, 545, 800),
                     "TECHNICAL DATA SHEET\n\n" + "\n".join(f"{k}: {v}" for k, v in SPECS) +
                     "\n\nSpare parts:\nArc chute assembly P/N 88123-A\nContact set P/N 88130-C\n"
                     "Motor drive unit P/N 90455-B", fontsize=13)
    pix = p.get_pixmap(dpi=140)
    out = fitz.open()
    op = out.new_page(width=p.rect.width, height=p.rect.height)
    op.insert_image(op.rect, stream=pix.tobytes("jpeg", jpg_quality=70))  # keep the file small
    out.save(OUT / "technical_data_scanned.pdf", deflate=True)

    # 4. Site memo (overrides the manual: shows pinned/site-specific answers)
    (OUT / "site_memo.md").write_text(
        "# Site memo 12 - Substation 4\n\n"
        "On Substation 4 the earth bolt torque is 45 Nm, not the 40 Nm printed in the manual, "
        "because of the retrofitted stainless bolts. Approved by the site engineer on 2 August 2026.\n"
    )
    print("manual corpus written to", OUT)


if __name__ == "__main__":
    main()
