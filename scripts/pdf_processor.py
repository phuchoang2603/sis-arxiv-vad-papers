import requests
import zipfile
import io
import shutil
from pathlib import Path

# Import all necessary config variables
from config import DOCLING_URL, DOCLING_HEADERS, DOCLING_PAYLOAD, DONE_DIR


def process_single_pdf(pdf_path: Path, base_output_dir: Path):
    """Processes a single PDF file."""

    print(f"Processing: {pdf_path.name}")
    file_stem = pdf_path.stem
    target_dir = base_output_dir / file_stem

    try:
        # 1. Create directory
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Directory: {target_dir}")

        # 2. Call API
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

            try:
                # Rename the .md file
                md_files = list(target_dir.glob("*.md"))
                if md_files:
                    original_md_path = md_files[0]
                    new_md_path = target_dir / "index.md"
                    original_md_path.rename(new_md_path)
                    print(f"  Renamed '{original_md_path.name}' to 'index.md'")
                else:
                    print(f"  WARNING: No .md file found in output to rename.")
            except Exception as e:
                print(f"  ERROR: Failed to rename .md file. {e}")

            # 4. Move the original file to the 'done' directory
            done_file_path = DONE_DIR / pdf_path.name
            shutil.move(pdf_path, done_file_path)
            print(f"  Moved original file to: {done_file_path}")

        else:
            print(
                f"  ERROR: API did not return a zip file. Got: {response.headers.get('Content-Type')}"
            )
            print(f"  Response text: {response.text[:200]}...")

    except requests.exceptions.RequestException as e:
        print(f"  ERROR: API call failed for {pdf_path.name}. {e}")
    except zipfile.BadZipFile:
        print(f"  ERROR: API returned a corrupt or invalid zip file.")
    except Exception as e:
        print(f"  ERROR: An unexpected error occurred. {e}")

    print("---")
