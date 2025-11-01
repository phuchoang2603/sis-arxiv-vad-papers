from config import INPUT_DIR, OUTPUT_DIR, DONE_DIR
from pdf_processor import process_single_pdf


def main():
    """Finds and loops through all PDFs, calling the processing function for each."""

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
