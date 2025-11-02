import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

INPUT_DIR = Path("/mnt/storage/media/docs/arxiv_existing")
OUTPUT_DIR = Path("/mnt/storage/media/docs/output/content/papers/")
DONE_DIR = Path("/mnt/storage/media/docs/arxiv_existing/done")

# --- API Configuration ---
DOCLING_URL = "http://localhost:5001/v1/convert/file"
N8N_WEBHOOK_URL = "https://localhost:5678/webhook/arxiv_extract_json_schema"
N8N_API_KEY = os.getenv("N8N_API_KEY")

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
