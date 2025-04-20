# Anthropic PDF Renamer (v1.0)

A robust, modular Python script for auto-renaming PDFs in a directory by extracting text, using Anthropic's Claude LLM to infer metadata (author, title, publication year), renaming files, and updating PDF metadata accordingly.

## Features

- **Batch renames PDFs** in a folder based on LLM-inferred metadata
- **Extracts text from the first 10 pages** (configurable) of each PDF (no OCR, text-based PDFs only)
- **Queries Anthropic Claude Haiku** for structured JSON metadata (author, title, pubdate)
- **Sanitizes and constructs safe filenames** (avoids OS issues, collisions, and length problems)
- **Updates internal PDF metadata** (title, author, creation date)
- **Extensible, testable, and robust**: modular design, clear separation of concerns, atomic file operations
- **Graceful error handling**: skips files on extraction or LLM errors, prints clear messages, never overwrites files
- **.env-based API key management** for secure configuration

## Usage

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Set up your `.env` file** in the project directory:
   ```
   ANTHROPIC_API_KEY=your_actual_api_key_here
   ```
3. **Run the script:**
   ```sh
   python claude-pdf-renamer-v1.0.py
   ```
   Enter the directory containing your PDFs when prompted.

## Notes & Best Practices
- Only works on text-based PDFs (no OCR support).
- Keeps your original files safe—never overwrites, always creates new files and removes originals only after successful processing.
- If the LLM returns unreliable metadata, the file is skipped and left unchanged.
- All configuration is done via `.env` for security.
- Designed for easy extension (swap out LLM, PDF library, or naming policy as needed).

## Caution

Please manage the security of your own data. Any loss of personal content due to errors in this script is your responsibility. Vet, test, and modify the script to meet your needs. Keep backups of important data. And of course, don't expose any files with sensitive data to commercial LLMs.

## Known Issues

- Will not work on image-based (OCR) PDFs; such files will be skipped or return 'Unknown' values.
- If LLM output is not close to valid JSON, the file will be skipped (but the script now robustly handles most common LLM output mistakes).

---

**Authors:** myPyAI + Naveen Srivatsav

_Last updated: 2024-06-04_
