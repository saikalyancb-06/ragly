"""Demo corpus for the Business persona: invoices spreadsheet, invoice PDFs, two contract versions.

Used by the eval suite and the demo. Everything is synthetic.
"""
from pathlib import Path

import pymupdf as fitz

OUT = Path(__file__).resolve().parent.parent / "samples" / "business"
OUT.mkdir(parents=True, exist_ok=True)

ROWS = [
    ("INV-1001", "2026-03-04", "ABC Ltd", "Substation upgrade", 62000, "Paid"),
    ("INV-1002", "2026-03-11", "ABC Ltd", "Spare parts", 48500, "Paid"),
    ("INV-1003", "2026-03-19", "Bharat Switchgear Pvt Ltd", "Breaker servicing", 125000, "Unpaid"),
    ("INV-1004", "2026-04-02", "ABC Ltd", "Annual maintenance", 90000, "Unpaid"),
    ("INV-1005", "2026-04-15", "Nova Instruments LLP", "Calibration", 27500, "Paid"),
    ("INV-1006", "2026-05-06", "Bharat Switchgear Pvt Ltd", "Emergency repair", 158000, "Paid"),
    ("INV-1007", "2026-05-21", "ABC Ltd", "Site inspection", 15000, "Paid"),
]
HEADERS = ["Invoice No", "Date", "Vendor", "Description", "Amount (INR)", "Status"]


def sheet() -> None:
    try:
        from openpyxl import Workbook
    except ImportError:
        print("openpyxl missing: writing CSV only")
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "Invoices"
    ws.append(HEADERS)
    for r in ROWS:
        ws.append(list(r))
    ws2 = wb.create_sheet("Monthly totals")
    ws2.append(["Month", "Expenditure (INR)", "Invoices"])
    for month, total, n in [("March 2026", 235500, 3), ("April 2026", 117500, 2), ("May 2026", 173000, 2)]:
        ws2.append([month, total, n])
    wb.save(OUT / "vendor_invoices.xlsx")


def csv_copy() -> None:
    lines = [",".join(HEADERS)] + [",".join(str(c) for c in r) for r in ROWS]
    (OUT / "vendor_invoices.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def invoice_pdf(row) -> None:
    no, date, vendor, desc, amount, status = row
    d = fitz.open()
    p = d.new_page()
    p.insert_textbox(fitz.Rect(50, 50, 545, 760),
                     f"TAX INVOICE\n\nInvoice No. {no}\nDate: {date}\n\nVendor: {vendor}\n"
                     f"Buyer: Kredo Automation Pvt Ltd, Bengaluru, Karnataka\n\n"
                     f"Description: {desc}\nAmount: INR {amount:,}\nGST (18%): INR {round(amount * 0.18):,}\n"
                     f"Total: INR {round(amount * 1.18):,}\nStatus: {status}\n\n"
                     f"Project: Substation 4 upgrade\nApproved by Mr. Ramesh Iyer, Project Manager at "
                     f"Kredo Automation Pvt Ltd.\nPayment terms: 30 days from invoice date.",
                     fontsize=11)
    d.save(OUT / f"invoice_{no}.pdf")


def contract(version: int) -> None:
    notice, fee, term, start = (60, "1,50,000", 24, "1 March 2026") if version == 1 else (90, "1,75,000", 36, "1 April 2026")
    extra = "" if version == 1 else ("\n\n6. INSURANCE\nThe Provider shall maintain professional indemnity "
                                     "insurance of INR 50,00,000 throughout the term.")
    d = fitz.open()
    p = d.new_page()
    p.insert_textbox(fitz.Rect(50, 50, 545, 780),
                     f"MASTER SERVICES AGREEMENT (v{version})\n\n"
                     f"This Agreement is made on {start} between ABC Ltd (Client) and "
                     f"Kredo Automation Pvt Ltd (Provider), Bengaluru.\n\n"
                     f"1. TERM\nThis Agreement continues for {term} months from {start}.\n\n"
                     f"2. FEES\nThe Client shall pay a monthly fee of INR {fee} plus GST within 15 days "
                     f"of invoice. Late payment attracts interest at 1.5% per month.", fontsize=11)
    p2 = d.new_page()
    p2.insert_textbox(fitz.Rect(50, 50, 545, 780),
                      f"3. TERMINATION\nEither party may terminate for convenience by giving {notice} days "
                      f"written notice. Material breach not cured within 30 days allows immediate termination.\n\n"
                      f"4. CONFIDENTIALITY\nObligations survive 5 years after termination.\n\n"
                      f"5. GOVERNING LAW\nThe laws of India apply and courts in Bengaluru have exclusive "
                      f"jurisdiction.{extra}", fontsize=11)
    d.save(OUT / f"service_agreement_v{version}.pdf")


if __name__ == "__main__":
    sheet()
    csv_copy()
    for r in ROWS[:3]:
        invoice_pdf(r)
    contract(1)
    contract(2)
    print("business corpus written to", OUT)
