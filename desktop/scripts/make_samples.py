"""Regenerate the demo documents in ./samples (contract, lab report, scanned note, markdown notes)."""
from pathlib import Path

import pymupdf as fitz

OUT = Path(__file__).resolve().parent.parent / "samples"
OUT.mkdir(exist_ok=True)


def pdf(name, pages):
    d = fitz.open()
    for t in pages:
        d.new_page().insert_textbox(fitz.Rect(50, 50, 550, 800), t, fontsize=11)
    d.save(OUT / name)


pdf("service_agreement.pdf", [
    "MASTER SERVICES AGREEMENT\n\nThis Agreement is made on 1 March 2026 between Acme Health Pvt Ltd (Client) and Kredo Automation (Provider).\n\n1. TERM\nThis Agreement starts on 1 March 2026 and continues for 24 months unless terminated earlier.\n\n2. FEES\nThe Client shall pay a monthly fee of INR 1,50,000 plus GST within 15 days of invoice. Late payments attract interest at 1.5% per month.",
    "3. TERMINATION\nEither party may terminate this Agreement for convenience by giving 60 days written notice. Either party may terminate immediately for material breach not cured within 30 days of notice.\n\n4. CONFIDENTIALITY\nAll patient data remains the property of the Client and must never leave the Client premises. Obligations survive for 5 years after termination.\n\n5. GOVERNING LAW\nThis Agreement is governed by the laws of India and courts in Bengaluru have exclusive jurisdiction.",
])
pdf("blood_report.pdf", [
    "CITY DIAGNOSTICS - LAB REPORT\n\nPatient: Ravi Kumar   Age: 42   Sample date: 12 August 2026\n\nHAEMATOLOGY\nHaemoglobin: 11.2 g/dL (reference 13.0 - 17.0) LOW\nWhite blood cells: 7,800 /uL (reference 4,000 - 11,000)\nPlatelets: 2.1 lakh /uL\n\nBIOCHEMISTRY\nFasting blood sugar: 132 mg/dL (reference 70 - 100) HIGH\nHbA1c: 6.9 %\nLDL cholesterol: 148 mg/dL\n\nRemarks: Findings suggest anaemia and poorly controlled blood sugar. Clinical correlation advised. Reviewed by Dr. S. Menon.",
])
src = fitz.open()
p = src.new_page()
p.insert_textbox(fitz.Rect(50, 50, 550, 800), "SITE INSPECTION NOTE\n\nInspection date: 3 September 2026\nInspector: Anita Rao\n\nThe north boundary wall has a crack of 2.5 metres.\nFire extinguishers in Block B expired in July 2026.\nRecommended action: replace extinguishers within 7 days.", fontsize=14)
pix = p.get_pixmap(dpi=150)
out = fitz.open()
op = out.new_page(width=p.rect.width, height=p.rect.height)
op.insert_image(op.rect, pixmap=pix)  # image only: forces the OCR path
out.save(OUT / "scanned_inspection.pdf")
(OUT / "team_notes.md").write_text("# Offsite notes\n\nThe team offsite is planned at Coorg on 10 October 2026. Budget approved is INR 3 lakh. Priya owns travel bookings.\n")
print("samples written to", OUT)
