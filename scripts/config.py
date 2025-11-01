from pathlib import Path

INPUT_DIR = Path("../../arxiv_existing/test")
OUTPUT_DIR = Path("../content/test")
DONE_DIR = Path("../../arxiv_existing/done")

# --- API Configuration ---
DOCLING_URL = "http://localhost:5001/v1/convert/file"

DOCLING_PAYLOAD = {
    "from_formats": "pdf",
    "to_formats": "md",
    "image_export_mode": "referenced",
    "do_ocr": "false",
    "ocr_engine": "rapidocr",
    "ocr_lang": '["english"]',
    "pdf_backend": "pypdfium2",
    "table_mode": "fast",
    "pipeline": "standard",
    "do_code_enrichment": "false",
    "do_formula_enrichment": "false",
    "target_type": "zip",
    "table_cell_matching": "false",
}

DOCLING_HEADERS = {"Accept": "application/json"}
