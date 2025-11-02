# SIS ArXiv VAD Papers

This repository contains the source code for the "SIS ArXiv VAD Papers" website, a [Hugo](https://gohugo.io/) static site using the [Blowfish](https://blowfish.page/) theme.

This project is a comprehensive platform for managing, processing, and displaying ArXiv research papers. It combines a Hugo static site with a powerful backend of containerized services for AI-driven PDF processing, metadata extraction, and ArXiv interaction.

![](https://i.ibb.co/chTK0Djn/image.png)

## Features

- **ArXiv AI Agent:** Includes an `mcp-arxiv-mcp-server`, which allows AI assistants to search, download, and read papers directly from the ArXiv repository.
- **Automated PDF-to-Markdown:** Uses the GPU-accelerated `docling-serve` to convert complex PDFs into clean Markdown.
- **AI Metadata Extraction:** A Python script orchestrates a pipeline that calls an `n8n` workflow to extract structured JSON metadata (title, authors, date, etc.) from converted text.
- **YAML Front Matter:** Automatically writes the extracted JSON back into the Markdown files as clean YAML front matter, making them ready to publish.
- **Hugo Static Site:** A clean, modern, and fast website built with Hugo and the Blowfish theme.

## Architecture & Services

The project's backend is defined in the `docker/compose.yml` file and includes several key services:

- **`n8n`:** The workflow automation service. It is used here as an API endpoint (via Webhook) to run the AI metadata extraction pipeline. It is also used to connect the `mcp-arxiv-mcp-server` to integrate with an LLM model for searching and downloading the latest papers.
- **`docling-serve`:** A powerful, GPU-enabled service that handles the core PDF-to-Markdown conversion. It is pre-loaded with models via the `docling-serve-initial` service.
- **`mcp-gateway` & `mcp-arxiv-mcp-server`:** A service that provides an AI-readable interface to the ArXiv repository, allowing for programmatic searching, downloading, and reading of papers.
- **Python Pipeline (`scripts/`):** This is the "glue" that connects everything. It is a host-run script that:
  1.  Finds new PDFs in an input directory.
  2.  Calls `docling-serve` to convert the PDF to Markdown.
  3.  Renames the output to `index.md` in a new `content/papers/` bundle.
  4.  Calls the `n8n` webhook with the path to the new `index.md`.
  5.  Receives the extracted JSON metadata back from n8n.
  6.  Writes this JSON as YAML front matter into the `index.md` file.

## File Structure

```
.
├── archetypes/         # Hugo new content templates
├── assets/             # Site assets (images, etc.)
├── config/             # Hugo configuration
├── content/            # The Markdown content for the site
│   └── papers/         # <-- Processed, AI-enhanced articles land here
├── docker/             # Docker service definitions
│   ├── compose.yml     # The main Docker Compose file for all services
│   └── catalog.yaml    # Describes the ArXiv MCP service
├── scripts/            # The Python automation pipeline
│   ├── config.py       # Holds paths and API configs
│   ├── main.py         # Main script to run the pipeline
│   ├── .env            # (Not shown) Stores secret keys
│   ├── pyproject.toml  # Python project definition
│   └── uv.lock         # Python dependencies
├── static/             # Static files (favicons, etc.)
├── themes/             # Hugo themes
│   └── blowfish/
└── hugo.toml           # Main Hugo configuration file
```

## Setup & Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/phuchoang2603/sis-arxiv-vad-papers.git
    cd sis-arxiv-vad-papers
    ```

2.  **Configure Docker Environment:**
    Create a `.env` file in the project's root directory (next to `docker/`). This will provide environment variables to your `compose.yml`.

    ```ini
    # ./.env

    # -- Docker Services --
    # MUST be an absolute path to your shared data folder
    SHARED_FOLDER=/path/to/your/shared/data

    # MUST be an absolute path for persistent Docker data
    APPDATA=/path/to/your/appdata/sis-arxiv

    # -- n8n --
    SUBDOMAIN=n8n
    DOMAIN_NAME=your-domain.com
    GENERIC_TIMEZONE=America/New_York
    ```

3.  **Configure n8n Workflow:**
    - Start your n8n instance and create your metadata extraction workflow.
    - **Start Node:** Use a **Webhook** node.
    - **Authentication:** Set to `Header Auth` and create a secure, random API key.
    - **Response Mode:** Set to `Respond at End of Workflow`. This is critical for getting the JSON response back.
    - **Workflow:** Add a `Read Binary File from Disk` node (using the path from the webhook), an `Extract from File` node, and your `Information Extractor` node.
    - **Activate:** Click the "Active" toggle in the top-right.
    - **Copy:** Copy the **Production URL**.

![](https://i.ibb.co/m5DxmYyc/image.png)

4.  **Configure Python Pipeline:**
    Create a separate `.env` file inside the `scripts/` directory for the Python script.

    ```ini
    # scripts/.env
    N8N_WEBHOOK_URL="https://n8n.your-domain.com/webhook/..." # <-- Your n8n PRODUCTION URL
    N8N_API_KEY="your-secret-n8n-header-auth-key"
    ```

5.  **Run Docker Services:**
    Run this command from the project's root directory:

    ```bash
    docker-compose -f docker/compose.yml up --build -d
    ```

    This will build and start `n8n`, `docling-serve`, and the other services.

6.  **Install Python Dependencies:**
    Navigate to the `scripts` directory and use `uv` to install:

    ```bash
    cd scripts
    uv sync
    ```

## How to Use the Pipeline

1.  **Add PDFs:** Place your `.pdf` files into the input directory defined in `scripts/config.py`. (By default, this points to `../../arxiv_existing/test`, which is a directory sibling to your project folder).
2.  **Run Pipeline:**
    ```bash
    cd scripts
    python main.py
    ```
3.  **Check Output:** Watch the terminal as the script processes each file. Your new content bundles, complete with `index.md` and YAML front matter, will appear in `content/papers/`.
4.  **Preview Site:**
    ```bash
    cd ..  # Return to the Hugo root
    hugo server
    ```
    Your site will be available at `http://localhost:1313`.

## License

This project is licensed under the MIT License.
