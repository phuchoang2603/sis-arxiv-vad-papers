# process_and_enhance.py

import requests
import zipfile
import io
import shutil
import json
from pathlib import Path
from ruamel.yaml import YAML

# Import all config variables
from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    DONE_DIR,
    DOCLING_URL,
    DOCLING_HEADERS,
    DOCLING_PAYLOAD,
    N8N_WEBHOOK_URL,
    N8N_API_KEY,
)


def extract_json_from_n8n(markdown_path: Path):
    """
    Sends a file path to the n8n workflow and returns the
    extracted JSON data.
    """
    print(f"  Sending '{markdown_path}' to n8n for extraction...")

    headers = {
        "x-api-key": N8N_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {"markdown_file_path": str(markdown_path)}

    try:
        response = requests.post(
            N8N_WEBHOOK_URL, data=json.dumps(payload), headers=headers, timeout=60
        )
        response.raise_for_status()

        response_data = response.json()

        if isinstance(response_data, dict) and "output" in response_data:
            print("  Successfully received JSON from n8n.")
            return response_data["output"]

        else:
            print(f"  ERROR: n8n response format is unexpected: {response_data}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"  ERROR: n8n workflow call failed: {e}")
        return None


def write_front_matter(md_file: Path, data: dict):
    """
    Writes the given dictionary as YAML front matter to the top
    of the markdown file.
    """
    try:
        # Read the existing content
        original_content = md_file.read_text(encoding="utf-8")

        # Initialize YAML writer
        yaml = YAML()
        yaml.preserve_quotes = True

        # Use an in-memory stream (io.StringIO) to write the YAML
        string_stream = io.StringIO()
        yaml.dump(data, string_stream)
        yaml_content = string_stream.getvalue()

        # Write the new content (front matter + original text)
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(yaml_content)
            f.write("---\n\n")  # Add a blank line for good formatting
            f.write(original_content)

        print(f"  Successfully wrote front matter to {md_file.name}")

    except Exception as e:
        print(f"  ERROR: Failed to write front matter: {e}")


def process_single_pdf(pdf_path: Path, base_output_dir: Path):
    """
    Full pipeline: Converts PDF, renames, calls n8n,
    and writes front matter.
    """
    print(f"Processing: {pdf_path.name}")
    file_stem = pdf_path.stem
    target_dir = base_output_dir / file_stem
    new_md_path = target_dir / "index.md"

    try:
        # 1. Create directory
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Directory: {target_dir}")

        # 2. Call Docling API
        with open(pdf_path, "rb") as f:
            files_payload = {"files": (pdf_path.name, f, "application/pdf")}
            print(f"  Sending to Docling API...")
            response = requests.post(
                DOCLING_URL,
                headers=DOCLING_HEADERS,
                data=DOCLING_PAYLOAD,
                files=files_payload,
                timeout=300,
            )
            response.raise_for_status()

        # 3. Unzip response
        if response.headers.get("Content-Type") == "application/zip":
            print(f"  Extracting zip file...")
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                zf.extractall(target_dir)

            # 4. Rename .md file
            md_file_path = new_md_path
            md_files = list(target_dir.glob("*.md"))
            md_files[0].rename(new_md_path)
            print(f"  Renamed '{md_files[0].name}' to 'index.md'")

            # 5. Call n8n and get JSON
            extracted_data = extract_json_from_n8n(md_file_path)

            # 6. Write JSON as YAML front matter
            if extracted_data:
                write_front_matter(md_file_path, extracted_data)
            else:
                print("  Skipping front matter, n8n call failed.")

            # 7. Move original PDF to 'done'
            done_file_path = DONE_DIR / pdf_path.name
            shutil.move(pdf_path, done_file_path)
            print(f"  Moved original file to: {done_file_path}")

        else:
            print(f"  ERROR: API did not return a zip file.")

    except Exception as e:
        print(f"  ERROR: Full processing failed for {pdf_path.name}. {e}")

    print("---")


def main():
    """Finds and loops through all PDFs."""

    print(f"Starting PDF processing...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Done directory: {DONE_DIR}")
    print("---")

    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found to process.")
        return

    for pdf_path in pdf_files:
        process_single_pdf(pdf_path, OUTPUT_DIR)


# --- Run the script ---
if __name__ == "__main__":
    main()
    print("All processing finished.")
