"""Tables: number parsing, header detection, extraction and deterministic answers."""
import pytest

from ragly_backend.tables import (
    MAX_SHEET_ROWS,
    answer_table_query,
    detect_numeric_cols,
    extract_pdf_tables,
    extract_sheet_tables,
    format_indian,
    is_table_question,
    parse_number,
    table_to_chunks,
    table_to_text,
)

# A small purchase register used by the query tests.
PURCHASES = {
    "id": 7,
    "doc_id": 3,
    "doc_name": "purchases.xlsx",
    "page": 2,
    "title": "Vendor Purchases",
    "headers": ["Invoice", "Vendor", "Month", "Qty", "Amount"],
    "rows": [
        ["INV-001", "ABC Traders", "March", "10", "1,50,000"],
        ["INV-002", "XYZ Supply", "March", "5", "75,000"],
        ["INV-003", "ABC Traders", "April", "8", "60,000"],
        ["INV-004", "ABC Traders", "March", "12", "85,000"],
        ["INV-005", "PQR Ltd", "May", "3", "40,000"],
    ],
    "numeric_cols": ["Qty", "Amount"],
}


# --------------------------------------------------------------- number parsing
@pytest.mark.parametrize("cell,expected", [
    ("1,50,000", 150000.0),          # Indian grouping
    ("150,000", 150000.0),           # Western grouping
    ("₹1,20,000.50", 120000.50),  # leading rupee symbol
    ("1,20,000.50 ₹", 120000.50),  # trailing symbol
    ("Rs. 4,500", 4500.0),
    ("INR 900", 900.0),
    ("(1,200)", -1200.0),            # accounting parentheses mean negative
    ("-1,200", -1200.0),
    ("12%", 12.0),                   # percent is a unit, not a division
    ("40 Nm", 40.0),                 # trailing unit dropped
    ("132 mg/dL", 132.0),
    ("2 lakh", 200000.0),
    ("1.5 crore", 15000000.0),
    ("0", 0.0),
    (".5", 0.5),
    ("1200", 1200.0),
])
def test_parse_number_accepts_indian_and_unit_formats(cell, expected):
    assert parse_number(cell) == pytest.approx(expected)


@pytest.mark.parametrize("cell", [
    "", " ", "-", "--", "—", "N/A", "n/a", "NA", "nil", "None", "TBD", "?",
    "abc", "approx 500", "1,200 2,300", "Q1", None,
])
def test_parse_number_rejects_non_numbers(cell):
    assert parse_number(cell) is None


def test_blank_is_none_not_zero():
    """A blank must not drag an average down, so it is None rather than 0.0."""
    assert parse_number("") is None and parse_number("0") == 0.0


def test_format_indian_grouping():
    assert format_indian(235000) == "2,35,000.00"
    assert format_indian(1500000) == "15,00,000.00"
    assert format_indian(999) == "999.00"
    assert format_indian(-1200.5) == "-1,200.50"


# ------------------------------------------------------ numeric column detection
def test_detect_numeric_cols_uses_60_percent_of_non_empty_cells():
    headers = ["Vendor", "Amount", "Mostly", "Notes"]
    rows = [
        ["ABC", "1,00,000", "10", "ok"],
        ["XYZ", "₹2,000", "20", "fine"],
        ["PQR", "-", "30", "good"],          # "-" is empty, not a failure
        ["LMN", "(500)", "text", "great"],
        ["OPQ", "3,000", "text", "nice"],
    ]
    numeric = detect_numeric_cols(headers, rows)
    assert "Amount" in numeric          # 4/4 non-empty cells parse
    assert "Mostly" in numeric          # 3/5 = 60%, exactly at the threshold
    assert "Vendor" not in numeric and "Notes" not in numeric


def test_detect_numeric_cols_ignores_empty_column():
    assert detect_numeric_cols(["A", "B"], [["", ""], ["", ""]]) == []


def test_detect_numeric_cols_below_threshold():
    rows = [["1"], ["2"], ["x"], ["y"]]      # 2/4 = 50%
    assert detect_numeric_cols(["Col"], rows) == []


# ---------------------------------------------------------- header detection
def test_first_row_is_used_as_header(tmp_path):
    csv_path = tmp_path / "invoices.csv"
    csv_path.write_text("Vendor,Amount\nABC Traders,1000\nXYZ,2000\n", encoding="utf-8")
    t = extract_sheet_tables(csv_path)[0]
    assert t["headers"] == ["Vendor", "Amount"]
    assert t["rows"] == [["ABC Traders", "1000"], ["XYZ", "2000"]]


def test_headers_are_synthesised_when_first_row_is_data(tmp_path):
    """A first row containing a number is data, so headers become col1..colN."""
    csv_path = tmp_path / "nohdr.csv"
    csv_path.write_text("ABC Traders,1000\nXYZ,2000\n", encoding="utf-8")
    t = extract_sheet_tables(csv_path)[0]
    assert t["headers"] == ["col1", "col2"]
    assert len(t["rows"]) == 2
    assert t["numeric_cols"] == ["col2"]


def test_blank_and_duplicate_header_cells_get_names(tmp_path):
    csv_path = tmp_path / "dup.csv"
    csv_path.write_text("Vendor,,Vendor\na,b,c\n", encoding="utf-8")
    t = extract_sheet_tables(csv_path)[0]
    assert t["headers"] == ["Vendor", "col2", "Vendor_2"]


# ------------------------------------------------------------- csv / tsv / xlsx
def test_csv_extraction_page_and_title(tmp_path):
    csv_path = tmp_path / "march_register.csv"
    csv_path.write_text("Vendor,Amount\nABC,1,000\n".replace("1,000", "1000"), encoding="utf-8")
    t = extract_sheet_tables(csv_path)[0]
    assert t["page"] == 1 and t["title"] == "march_register"


def test_tsv_delimiter_is_detected(tmp_path):
    tsv = tmp_path / "data.tsv"
    tsv.write_text("Vendor\tAmount\nABC Traders\t1,50,000\nXYZ\t75,000\n", encoding="utf-8")
    t = extract_sheet_tables(tsv)[0]
    assert t["headers"] == ["Vendor", "Amount"]
    assert t["rows"][0] == ["ABC Traders", "1,50,000"]
    assert t["numeric_cols"] == ["Amount"]


def test_csv_drops_fully_empty_rows_and_columns(tmp_path):
    csv_path = tmp_path / "gappy.csv"
    csv_path.write_text("Vendor,,Amount\nABC,,1000\n,,\nXYZ,,2000\n", encoding="utf-8")
    t = extract_sheet_tables(csv_path)[0]
    assert t["headers"] == ["Vendor", "Amount"]        # the blank column is gone
    assert t["rows"] == [["ABC", "1000"], ["XYZ", "2000"]]   # the blank row is gone


def test_xlsx_extraction_one_table_per_sheet(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "book.xlsx"
    book = openpyxl.Workbook()
    first = book.active
    first.title = "Purchases"
    first.append(["Vendor", "Month", "Amount"])
    first.append(["ABC Traders", "March", 150000])
    first.append(["XYZ Supply", "March", 75000])
    second = book.create_sheet("Summary")
    second.append(["Metric", "Value"])
    second.append(["Invoices", 2])
    book.save(path)

    tables = extract_sheet_tables(path)
    assert [t["title"] for t in tables] == ["Purchases", "Summary"]
    assert [t["page"] for t in tables] == [1, 2]
    assert tables[0]["headers"] == ["Vendor", "Month", "Amount"]
    # openpyxl reads whole numbers as floats: they must not arrive as "150000.0"
    assert tables[0]["rows"][0] == ["ABC Traders", "March", "150000"]
    assert tables[0]["numeric_cols"] == ["Amount"]


def test_xlsx_row_cap(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "big.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["Amount"])
    for i in range(MAX_SHEET_ROWS + 200):
        sheet.append([i])
    book.save(path)
    t = extract_sheet_tables(path)[0]
    assert len(t["rows"]) <= MAX_SHEET_ROWS


def test_unsupported_spreadsheet_type_raises(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_sheet_tables(bad)


# ------------------------------------------------------------- pdf extraction
def _make_table_pdf(path):
    """A one-page PDF with a real ruled table.

    PyMuPDF only grew `Page.insert_table` in later builds, so when it is missing the
    same table is drawn by hand - grid lines plus positioned text - which is exactly
    what `find_tables` reads anyway.
    """
    import pymupdf

    rows = [
        ["Vendor", "Month", "Amount"],
        ["ABC Traders", "March", "1,50,000"],
        ["XYZ Supply", "March", "75,000"],
        ["ABC Traders", "April", "60,000"],
    ]
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((60, 62), "Vendor Purchase Summary", fontsize=13)
    if hasattr(page, "insert_table"):        # pragma: no cover - newer PyMuPDF only
        page.insert_table(pymupdf.Rect(60, 90, 400, 186), rows)
    else:
        widths = [140, 90, 110]
        x0, y0, height = 60, 90, 24
        xs = [x0]
        for w in widths:
            xs.append(xs[-1] + w)
        for i in range(len(rows) + 1):
            page.draw_line((x0, y0 + i * height), (xs[-1], y0 + i * height))
        for x in xs:
            page.draw_line((x, y0), (x, y0 + len(rows) * height))
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                page.insert_text((xs[c] + 4, y0 + r * height + 16), cell, fontsize=10)
    doc.save(path)
    doc.close()
    return rows


def test_pdf_table_extraction(tmp_path):
    pytest.importorskip("pymupdf")
    path = tmp_path / "purchases.pdf"
    _make_table_pdf(path)
    tables = extract_pdf_tables(path)
    assert len(tables) == 1
    t = tables[0]
    assert t["page"] == 1
    assert t["headers"] == ["Vendor", "Month", "Amount"]
    assert t["rows"][0] == ["ABC Traders", "March", "1,50,000"]
    assert len(t["rows"]) == 3
    assert t["numeric_cols"] == ["Amount"]
    # the caption above the table is picked up as the title
    assert t["title"] == "Vendor Purchase Summary"


def test_pdf_with_no_table_yields_nothing(tmp_path):
    import pymupdf

    path = tmp_path / "prose.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "This agreement may be terminated on 60 days notice.")
    doc.save(path)
    doc.close()
    assert extract_pdf_tables(path) == []


def test_extracted_pdf_table_can_be_answered(tmp_path):
    """Extraction and answering join up end to end."""
    path = tmp_path / "purchases.pdf"
    _make_table_pdf(path)
    tables = extract_pdf_tables(path)
    answer = answer_table_query("What is the total amount from ABC Traders?", tables)
    assert answer is not None
    assert answer["operation"] == "sum" and answer["column"] == "Amount"
    assert answer["value"] == pytest.approx(210000.0)


# ------------------------------------------------------------- text rendering
def test_table_to_text_includes_title_and_headers():
    text = table_to_text(PURCHASES)
    assert "Vendor Purchases" in text
    assert "| Vendor | Month | Qty | Amount |" in text
    assert "ABC Traders" in text
    assert len(text) <= 1500


def test_small_table_is_a_single_chunk():
    chunks = table_to_chunks(PURCHASES)
    assert len(chunks) == 1 and chunks[0] == table_to_text(PURCHASES)


def test_big_table_splits_and_repeats_the_header():
    big = dict(PURCHASES)
    big["rows"] = [[f"INV-{i:04d}", "ABC Traders", "March", "1", "1,000"] for i in range(400)]
    chunks = table_to_chunks(big)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 1500
        assert "| Vendor | Month | Qty | Amount |" in chunk   # header repeated
        assert "Vendor Purchases" in chunk
    # every row survives the split
    body = sum(len([ln for ln in c.splitlines() if ln.startswith("| INV-")]) for c in chunks)
    assert body == 400
    # the single-chunk form says how much it left out
    assert "more rows" in table_to_text(big)


def test_table_with_no_rows_still_renders():
    text = table_to_text({"page": 1, "title": "Empty", "headers": ["A", "B"], "rows": []})
    assert "Empty" in text and "| A | B |" in text


# ------------------------------------------------------------ query answering
def test_sum_with_vendor_filter():
    a = answer_table_query("What is the total amount from ABC Traders?", [PURCHASES])
    assert a is not None
    assert a["operation"] == "sum" and a["column"] == "Amount"
    assert a["value"] == pytest.approx(295000.0)
    assert a["formatted"] == "₹2,95,000.00"      # Indian grouping, money column
    assert a["row_count"] == 3
    assert [f["column"] for f in a["filters"]] == ["Vendor"]
    assert a["filters"][0]["op"] == "contains"
    assert {r["Invoice"] for r in a["matched_rows"]} == {"INV-001", "INV-003", "INV-004"}
    assert "sum(Amount) over 3 rows where Vendor contains" in a["workings"]
    assert a["confidence"] == "high"
    # the source keys of the table ride along for citation
    assert (a["table_id"], a["doc_id"], a["doc_name"], a["page"]) == (7, 3, "purchases.xlsx", 2)
    assert a["title"] == "Vendor Purchases"


def test_sum_with_month_filter():
    a = answer_table_query("What is the total spend in March?", [PURCHASES])
    assert a is not None
    assert a["operation"] == "sum" and a["column"] == "Amount"
    assert a["value"] == pytest.approx(310000.0)         # 150000 + 75000 + 85000
    assert a["row_count"] == 3
    assert a["filters"] == [{"column": "Month", "op": "month", "value": "March"}]
    assert "Month = 'March'" in a["workings"]


def test_sum_with_vendor_and_month_filter():
    a = answer_table_query("How much did we spend with ABC Traders in March?", [PURCHASES])
    assert a is not None
    assert a["value"] == pytest.approx(235000.0)          # 150000 + 85000
    assert a["formatted"] == "₹2,35,000.00"
    assert {f["column"] for f in a["filters"]} == {"Vendor", "Month"}


def test_how_much_did_we_pay_phrasing():
    """"pay"/"spend"/"paid" all have to reach the money column."""
    a = answer_table_query("How much did we pay XYZ Supply?", [PURCHASES])
    assert a is not None
    assert a["operation"] == "sum" and a["column"] == "Amount"
    assert a["value"] == pytest.approx(75000.0)
    assert a["matched_rows"] == [dict(zip(PURCHASES["headers"], PURCHASES["rows"][1]))]


def test_calendar_quarter_filter():
    a = answer_table_query("What is the total amount in Q1?", [PURCHASES])
    assert a is not None
    assert a["filters"][0] == {"column": "Month", "op": "quarter", "value": "Q1"}
    assert a["value"] == pytest.approx(310000.0)      # Q1 is Jan-Mar, so the March rows


def test_average():
    a = answer_table_query("What is the average amount per invoice?", [PURCHASES])
    assert a is not None
    assert a["operation"] == "avg" and a["column"] == "Amount"
    assert a["value"] == pytest.approx(82000.0)           # 410000 / 5
    assert a["row_count"] == 5


def test_count_with_vendor_filter():
    a = answer_table_query("How many invoices from ABC Traders?", [PURCHASES])
    assert a is not None
    assert a["operation"] == "count"
    assert a["value"] == pytest.approx(3.0)
    assert a["row_count"] == 3 and a["formatted"] == "3 rows"
    assert [f["column"] for f in a["filters"]] == ["Vendor"]


def test_max_returns_the_winning_row_as_evidence():
    a = answer_table_query("Which vendor has the highest amount?", [PURCHASES])
    assert a is not None
    assert a["operation"] == "max" and a["value"] == pytest.approx(150000.0)
    assert len(a["matched_rows"]) == 1
    assert a["matched_rows"][0]["Vendor"] == "ABC Traders"
    assert a["matched_rows"][0]["Invoice"] == "INV-001"


def test_min_with_month_filter():
    a = answer_table_query("What is the lowest amount in March?", [PURCHASES])
    assert a is not None
    assert a["operation"] == "min" and a["value"] == pytest.approx(75000.0)
    assert a["matched_rows"][0]["Invoice"] == "INV-002"


def test_numeric_above_filter():
    a = answer_table_query("What is the total amount above 50000?", [PURCHASES])
    assert a is not None
    assert a["operation"] == "sum"
    assert {"column": "Amount", "op": ">", "value": 50000.0} in a["filters"]
    assert a["value"] == pytest.approx(370000.0)          # every row except the 40,000 one
    assert a["row_count"] == 4


def test_numeric_filter_accepts_rupee_formatting():
    a = answer_table_query("total amount more than ₹50,000", [PURCHASES])
    assert a is not None and a["value"] == pytest.approx(370000.0)


def test_between_filter():
    a = answer_table_query("How many invoices have quantity between 5 and 10?", [PURCHASES])
    assert a is not None
    assert a["operation"] == "count" and a["row_count"] == 3    # qty 10, 5, 8
    assert a["filters"][0]["op"] == "between"
    assert a["filters"][0]["value"] == [5.0, 10.0]


def test_pure_lookup_has_no_value():
    a = answer_table_query("List invoices from ABC above 50000", [PURCHASES])
    assert a is not None
    assert a["operation"] == "filter"
    assert "value" not in a
    assert a["row_count"] == 3
    assert {f["op"] for f in a["filters"]} == {">", "contains"}
    assert all(r["Vendor"] == "ABC Traders" for r in a["matched_rows"])


def test_quantity_column_is_not_formatted_as_money():
    a = answer_table_query("What is the total qty from ABC Traders?", [PURCHASES])
    assert a is not None
    assert a["column"] == "Qty" and a["value"] == pytest.approx(30.0)
    assert a["formatted"] == "30"                    # plain number, no rupee sign


def test_difference_is_max_minus_min():
    a = answer_table_query("What is the difference in amount?", [PURCHASES])
    assert a is not None
    assert a["operation"] == "diff"
    assert a["value"] == pytest.approx(110000.0)     # 150000 - 40000
    assert "max(Amount) - min(Amount)" in a["workings"]


def test_iso_date_column_month_filter():
    t = {
        "page": 1, "title": "Invoices", "headers": ["Invoice Date", "Vendor", "Amount"],
        "rows": [
            ["2026-03-04", "ABC Traders", "1,00,000"],
            ["2026-03-29", "ABC Traders", "50,000"],
            ["2026-04-02", "ABC Traders", "70,000"],
        ],
    }
    a = answer_table_query("total amount in March 2026", [t])
    assert a is not None
    assert a["value"] == pytest.approx(150000.0)
    assert a["filters"][0] == {"column": "Invoice Date", "op": "month", "value": "March 2026"}


def test_fiscal_year_filter_runs_april_to_march():
    t = {
        "page": 1, "title": "Invoices", "headers": ["Date", "Amount"],
        "rows": [
            ["2025-04-10", "1,000"],      # in FY26
            ["2026-03-31", "2,000"],      # in FY26
            ["2026-04-01", "4,000"],      # FY27
            ["2025-03-31", "8,000"],      # FY25
        ],
    }
    a = answer_table_query("what is the total amount in FY26?", [t])
    assert a is not None
    assert a["value"] == pytest.approx(3000.0)
    assert a["filters"][0]["op"] == "fy"


def test_most_relevant_table_is_chosen():
    other = {
        "id": 1, "page": 1, "title": "Torque Settings",
        "headers": ["Fastener", "Torque"], "rows": [["M8", "40 Nm"], ["M10", "80 Nm"]],
    }
    a = answer_table_query("What is the total amount from ABC Traders?", [other, PURCHASES])
    assert a is not None and a["table_id"] == 7 and a["column"] == "Amount"


# ----------------------------------------------------------------- refusals
def test_non_table_question_returns_none():
    """The contract question in the sample set must never be answered from a table."""
    assert answer_table_query("What is the termination notice period?", [PURCHASES]) is None


@pytest.mark.parametrize("question", [
    "What is the termination notice period?",
    "Summarise the agreement",
    "Who signed the contract?",
])
def test_questions_with_no_arithmetic_return_none(question):
    assert answer_table_query(question, [PURCHASES]) is None


def test_no_tables_returns_none():
    assert answer_table_query("What is the total amount?", []) is None


def test_ambiguous_column_returns_none():
    """Two numeric columns, both named by the question: refuse rather than guess."""
    t = {
        "page": 1, "title": "Rates", "headers": ["Item", "Amount", "Cost"],
        "rows": [["Bolt", "10", "5"], ["Nut", "20", "6"]],
    }
    assert answer_table_query("what is the total amount and cost?", [t]) is None


def test_named_column_wins_over_a_weaker_match():
    """Ambiguity is refused, but a clearly named column is still resolved."""
    t = {
        "page": 1, "title": "Rates", "headers": ["Item", "Unit Price", "Units"],
        "rows": [["Bolt", "10", "5"], ["Nut", "20", "6"]],
    }
    a = answer_table_query("what is the average unit price?", [t])
    assert a is not None and a["column"] == "Unit Price"
    assert a["value"] == pytest.approx(15.0)


def test_no_numeric_column_returns_none():
    t = {
        "page": 1, "title": "Contacts", "headers": ["Name", "City"],
        "rows": [["Asha", "Pune"], ["Ravi", "Delhi"]],
    }
    assert answer_table_query("What is the total spend?", [t]) is None


def test_unmatched_filter_returns_none():
    """No matching row is more often a mis-read filter than a true zero."""
    assert answer_table_query("What is the total amount from LMN Corp?", [PURCHASES]) is None


def test_comparison_with_no_identifiable_column_returns_none():
    t = {
        "page": 1, "title": "Readings", "headers": ["Sensor", "Torque", "Pressure"],
        "rows": [["S1", "40", "10"], ["S2", "80", "20"]],
    }
    assert answer_table_query("how many readings above 15?", [t]) is None


# --------------------------------------------------------------- store round-trip
def test_extracted_table_survives_the_store(tmp_path):
    """Extraction -> replace_tables -> list_tables -> answer, with citations intact."""
    from ragly_backend.store import Store

    csv_path = tmp_path / "purchases.csv"
    csv_path.write_text(
        'Vendor,Month,Amount\n'
        'ABC Traders,March,"1,50,000"\n'
        'XYZ Supply,March,"75,000"\n'
        'ABC Traders,April,"60,000"\n',
        encoding="utf-8",
    )
    store = Store(tmp_path / "tables.db")
    doc_id = store.add_document("purchases.csv", str(csv_path), "sha-tables-test", 100, 1)
    store.replace_tables(doc_id, extract_sheet_tables(csv_path))
    store.mark_ready(doc_id)

    stored = store.list_tables()
    assert len(stored) == 1
    assert stored[0]["headers"] == ["Vendor", "Month", "Amount"]
    assert stored[0]["numeric_cols"] == ["Amount"]

    a = answer_table_query("What is the total amount from ABC Traders?", stored)
    assert a is not None
    assert a["value"] == pytest.approx(210000.0)
    # the row ids the store assigned are carried through for citation
    assert a["table_id"] == stored[0]["id"]
    assert a["doc_id"] == doc_id and a["doc_name"] == "purchases.csv"


# ------------------------------------------------------------------- router
@pytest.mark.parametrize("question", [
    "What is the total spend?",
    "What is the average amount?",
    "How much did we pay ABC Traders?",
    "How many invoices are there?",
    "count of invoices",
    "Which vendor has the highest amount?",
    "What was the lowest price?",
    "What did we spend in March?",
    "invoices above 50000",
    "compare amounts between vendors",
    "total expenditure in FY26",
])
def test_is_table_question_true(question):
    assert is_table_question(question) is True


@pytest.mark.parametrize("question", [
    "What is the termination notice period?",
    "Summarise the agreement",
    "Who signed the contract?",
    "What does error E-47 mean?",
    "",
])
def test_is_table_question_false(question):
    assert is_table_question(question) is False


# --------------------------------------------------- real-corpus regressions (bank statement)
BANK_HEADERS = ["Date", "Description", "Debit", "Credit", "Category"]
BANK_ROWS = [
    ["03 Sep", "NEFT-MERIDIAN-CAPITAL-EMI", "57,530", "—", "Loan EMI"],
    ["05 Sep", "AWS CLOUD SERVICES", "18,400", "—", "Software"],
    ["07 Sep", "SALARY CREDIT", "—", "6,20,000", "Payroll"],
    ["10 Sep", "OFFICE LEASE - BENGALURU", "85,000", "—", "Rent"],
    ["13 Sep", "TRAVEL DESK - MYSURU", "12,650", "—", "Travel"],
    ["16 Sep", "CARD PURCHASE - CAMERA STORE", "46,900", "—", "Equipment"],
    ["20 Sep", "MERIDIAN CAPITAL - PREPAYMENT", "2,00,000", "—", "Loan Prepayment"],
    ["24 Sep", "CLIENT PAYMENT - BLUEPEAK", "—", "3,40,000", "Receivable"],
    ["28 Sep", "BANK SERVICE CHARGE", "590", "—", "Bank Charges"],
]


def _bank_table():
    from ragly_backend.tables import detect_numeric_cols

    return {"table_id": 1, "doc_id": 1, "doc_name": "corpus.pdf", "page": 4, "title": "Bank Transaction Extract",
            "headers": BANK_HEADERS, "rows": BANK_ROWS,
            "numeric_cols": detect_numeric_cols(BANK_HEADERS, BANK_ROWS)}


def test_a_dash_is_an_empty_cell_not_a_non_number():
    """Credit is mostly em-dashes with two real amounts: it is still a numeric column."""
    from ragly_backend.tables import detect_numeric_cols

    cols = detect_numeric_cols(BANK_HEADERS, BANK_ROWS)
    assert "Debit" in cols and "Credit" in cols
    assert "Date" not in cols          # "03 Sep" is a date, not a quantity


def test_total_debit():
    from ragly_backend.tables import answer_table_query

    res = answer_table_query("what is the total debit", [_bank_table()])
    assert res and res["column"] == "Debit" and res["value"] == 421070


def test_total_credit_does_not_filter_on_the_column_name():
    """Regression: "credit" matched the text "SALARY CREDIT" and silently dropped a row."""
    from ragly_backend.tables import answer_table_query

    res = answer_table_query("total credit", [_bank_table()])
    assert res and res["column"] == "Credit"
    assert res["value"] == 960000, "both credit rows must be counted"
    assert not res["filters"], "the column name must not become a row filter"


def test_a_word_stem_finds_the_column():
    from ragly_backend.tables import answer_table_query

    res = answer_table_query("how much was credited", [_bank_table()])
    assert res and res["column"] == "Credit"


def test_a_sentence_fragment_is_not_used_as_a_table_title():
    from ragly_backend.tables import _heading_ish

    assert _heading_ish("Indexed asset inventory")
    assert _heading_ish("Severity Matrix")
    assert not _heading_ish("require exact retrieval, some require synthesis, and some should trigger abstention.")
    assert not _heading_ish("grounding, citations and abstention.")


def test_a_stale_stored_column_list_does_not_decide_the_answer():
    """An older index recorded numeric_cols without Credit; the live rows must win."""
    from ragly_backend.tables import answer_table_query

    stale = _bank_table()
    stale["numeric_cols"] = ["Date", "Debit"]          # exactly what the old build stored
    res = answer_table_query("find the total credit amount in page 4", [stale])
    assert res and res["column"] == "Credit" and res["value"] == 960000


def test_a_total_is_grouped_like_the_cells_it_came_from():
    from ragly_backend.tables import answer_table_query

    res = answer_table_query("total debit", [_bank_table()])
    assert res["formatted"].startswith("4,21,070"), res["formatted"]


# --------------------------------------------------- asking for a value, not a row count
ASSETS = {"table_id": 9, "doc_id": 1, "doc_name": "corpus.pdf", "page": 7, "title": "Indexed asset inventory",
          "headers": ["Asset class", "Count", "Primary processing"],
          "rows": [["Documents", "126", "Text parsing + embeddings"],
                   ["Images", "84", "Visual embeddings + OCR"],
                   ["Tables", "57", "Structure-aware parsing"],
                   ["Scanned pages", "39", "OCR + layout recovery"],
                   ["Other", "24", "Metadata extraction"]],
          "numeric_cols": ["Count"]}

META = {"table_id": 10, "doc_id": 1, "doc_name": "corpus.pdf", "page": 11, "title": "RAG Evaluation Questions",
        "headers": ["#", "Question", "Expected behavior"],
        "rows": [[str(i), f"Question {i} about the invoice", "Exact answer + page citation"]
                 for i in range(1, 6)],
        "numeric_cols": ["#"]}


def test_a_column_called_count_is_read_not_counted():
    """Regression: "what is the count of documents" answered "1 row" instead of 126."""
    from ragly_backend.tables import answer_table_query

    res = answer_table_query("what is the count of documents", [ASSETS])
    assert res and res["value"] == 126 and res["operation"] == "lookup"


def test_how_many_of_a_category_reads_that_row():
    from ragly_backend.tables import answer_table_query

    res = answer_table_query("how many scanned pages", [ASSETS])
    assert res and res["value"] == 39


def test_a_row_number_column_is_never_summed():
    """A column of 1, 2, 3 … is an index. Summing it produced answers like "4 is the total #"."""
    from ragly_backend.tables import detect_numeric_cols, answer_table_query

    assert detect_numeric_cols(META["headers"], META["rows"]) == []
    assert answer_table_query("what is the invoice number", [META], strict=True) is None


def test_a_name_in_a_cell_does_not_make_a_numeric_answer():
    """"Who is Ananya Rao?" must fall through to retrieval, not return a number."""
    from ragly_backend.tables import answer_table_query

    rows = [["1", "Who is Ananya Rao?", "Exact answer"]]
    table = {**META, "rows": rows}
    assert answer_table_query("Who is Ananya Rao?", [table], strict=True) is None


def test_a_dated_row_is_looked_up_by_its_label():
    from ragly_backend.tables import answer_table_query

    res = answer_table_query("what was debited on 16 Sep", [_bank_table()])
    assert res and res["value"] == 46900


# --------------------------------------------------- measured values (units, not just numbers)
PLANTS = {"table_id": 20, "doc_id": 2, "doc_name": "control.pdf", "page": 5,
          "title": "Plant Observation Log",
          "headers": ["Date", "Plant", "Height", "Observation"],
          "rows": [["2 Sep 2026", "Basil", "18 cm", "New leaves visible"],
                   ["5 Sep 2026", "Tomato", "31 cm", "One flower cluster"],
                   ["8 Sep 2026", "Mint", "14 cm", "Healthy foliage"],
                   ["11 Sep 2026", "Spinach", "9 cm", "Several new leaves"],
                   ["14 Sep 2026", "Basil", "21 cm", "Strong new growth"]],
          "numeric_cols": []}


def test_a_value_with_a_unit_is_a_number_a_date_is_not():
    from ragly_backend.tables import _is_quantity, detect_numeric_cols

    assert _is_quantity("18 cm") and _is_quantity("9.25%") and _is_quantity("₹57,530")
    assert not _is_quantity("03 Sep") and not _is_quantity("March 2026")
    assert detect_numeric_cols(PLANTS["headers"], PLANTS["rows"]) == ["Height"]


def test_asking_for_a_measurement_returns_it_with_its_unit():
    """Regression: "what is the height of basil" answered "2 rows"."""
    from ragly_backend.tables import answer_table_query

    res = answer_table_query("what is the height of basil", [PLANTS])
    assert res and res["operation"] == "lookup"
    assert res["formatted"] == "18 cm and 21 cm", res["formatted"]


def test_measurements_are_listed_not_added():
    from ragly_backend.tables import answer_table_query

    res = answer_table_query("what is the height of mint", [PLANTS])
    assert res["formatted"] == "14 cm"


def test_superlatives_work_and_keep_the_unit():
    from ragly_backend.tables import answer_table_query

    assert answer_table_query("tallest plant", [PLANTS])["formatted"] == "31 cm"
    assert answer_table_query("shortest plant", [PLANTS])["formatted"] == "9 cm"


def test_a_running_header_is_removed_from_page_text():
    from ragly_backend.pipeline import strip_running_headers

    pages = [(i, f"Acme Quarterly Report · Confidential Page {i}\nSection {i} says something "
                 f"specific about topic {i}.\nAnother distinct line {i}.") for i in range(1, 8)]
    out = strip_running_headers(pages)
    assert all("Acme Quarterly Report" not in text for _page, text in out)
    assert all(f"Section {page} says" in text for page, text in out)


def test_a_blank_cell_answers_zero_rather_than_one_row():
    """Regression: "how many labour??" on a parts table answered "1 row". The row exists and
    its quantity cell is a dash, so the answer is 0."""
    from ragly_backend.tables import answer_table_query

    table = {"id": 7, "doc_id": 1, "doc_name": "corpus.pdf", "page": 8, "title": "Repair Parts Schedule",
             "headers": ["Part", "Qty", "Unit", "Total", "Status"],
             "rows": [["Front-right bumper cover", "1", "₹24,800", "₹24,800", "Ordered"],
                      ["Right headlamp assembly", "1", "₹31,500", "₹31,500", "Pending"],
                      ["Clips and fasteners", "1 set", "₹1,650", "₹1,650", "In stock"],
                      ["Paint and finishing", "1", "₹9,500", "₹9,500", "Scheduled"],
                      ["Labour", "—", "₹5,000", "₹5,000", "Scheduled"],
                      ["TOTAL", "—", "—", "₹72,450", "—"]]}

    blank = answer_table_query("how many labour??", [table])
    assert blank["formatted"] == "0"
    assert "blank" in blank["workings"]

    present = answer_table_query("how many paint and finishing", [table])
    assert present["formatted"] == "1"


def test_the_tables_own_total_row_is_not_added_into_the_total():
    """Regression: summing a column that ends in a TOTAL row counted every amount twice
    (72,450 printed, 1,44,900 answered)."""
    from ragly_backend.tables import answer_table_query

    table = {"id": 7, "doc_id": 1, "doc_name": "corpus.pdf", "page": 8, "title": "Repair Parts Schedule",
             "headers": ["Part", "Qty", "Unit", "Total", "Status"],
             "rows": [["Front-right bumper cover", "1", "₹24,800", "₹24,800", "Ordered"],
                      ["Right headlamp assembly", "1", "₹31,500", "₹31,500", "Pending"],
                      ["Clips and fasteners", "1 set", "₹1,650", "₹1,650", "In stock"],
                      ["Paint and finishing", "1", "₹9,500", "₹9,500", "Scheduled"],
                      ["Labour", "—", "₹5,000", "₹5,000", "Scheduled"],
                      ["TOTAL", "—", "—", "₹72,450", "—"]]}

    res = answer_table_query("what is the total", [table])
    assert res["value"] == 72450.0
    assert "excluding the table's own TOTAL row" in res["workings"]
    assert answer_table_query("how many parts are there", [table])["value"] == 5.0
