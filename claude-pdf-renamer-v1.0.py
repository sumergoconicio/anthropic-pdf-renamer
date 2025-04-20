########### PYTHON
# Script Title: Anthropic PDF Renamer (v0.6 - Refactored)
# Script Description: Auto-renames PDFs in a directory by extracting first-page text, leveraging Anthropic's Claude LLM to infer author/title/pubdate, renaming files, and updating PDF metadata accordingly. Rewrites, extension, and clarity improvements per modern Python and systems best practice.
# Script Author: myPyAI + Naveen Srivatsav
# Last Updated: 20240604
# TLA+ Abstract:
# """
# tla ---- MODULE pdf_renamer ----
# VARIABLES pdf_files, extracted_text, ai_metadata, renamed_files
# Init == pdf_files \in Directory /\ extracted_text = <<>> /\ ai_metadata = <<>> /\ renamed_files = {}
# Next == \E file \in pdf_files:
#      /\ extracted_text' = extract_first_page(file)
#      /\ ai_metadata' = query_llm(extracted_text)
#      /\ IF Valid(ai_metadata') THEN
#            renamed_files' = Renamed(renamed_files, file, ai_metadata')
#         ELSE
#            renamed_files' = Unchanged(renamed_files, file)
# Success == \A f \in pdf_files: FileIsNamedAndTagged(f)
# ----
# """
# Major changes: 
# - Further modularization for clarity and testability
# - All I/O and side-effects managed in main()
# - More robust error handling and reporting
# - Clean separation of LLM querying, text extraction, metadata processing, and renaming
# - Extensible design; easy to plug in other LLMs, PDF libraries, or naming policies
# - Uses pathlib for safer path operations and better cross-OS support
###########

# To run: pip install anthropic PyPDF2 python-dotenv
import os  # Standard: Directory listing, env var fallback
import re  # Standard: String sanitization
import json  # Standard: Parsing LLM responses, file-safe serialization
from pathlib import Path  # Standard: Modern path handling replaces os.path
from typing import Optional, Dict  # Standard: Type hints for maintainability

# Third-party: Deep dependency
from anthropic import Anthropic         # LLM API client
from PyPDF2 import PdfReader, PdfWriter # PDF operations
from dotenv import load_dotenv          # Secure .env configuration

# INGREDIENTS:
# - Standard: os, re, json, pathlib, typing
# - Third-party: anthropic, PyPDF2, python-dotenv

def load_llm(api_key_env_var: str = "ANTHROPIC_API_KEY") -> Anthropic:
    """
    Big-picture: Initialize and return Anthropic client using secure API key lookup, with .env fallback.
    Inputs: api_key_env_var - name of environment variable containing API key.
    Outputs: Anthropic client object.
    Role: Dependency management and setup.
    """
    load_dotenv()
    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise RuntimeError("Anthropic API key is missing. Please set the ANTHROPIC_API_KEY environment variable or .env file.")
    return Anthropic(api_key=api_key)

def extract_first_n_pages_text(pdf_path: Path, n: int = 10) -> Optional[str]:
    """
    Big-picture: Extract text content from the first n pages of a PDF (no OCR, only text PDF).
    Inputs: pdf_path - Path to PDF file; n - number of pages to extract.
    Outputs: Extracted text string, or None on failure.
    Role: Provides "raw" string for LLM analysis.
    """
    try:
        reader = PdfReader(str(pdf_path))
        if not reader.pages:
            return None
        num_pages = min(len(reader.pages), n)
        texts = []
        for i in range(num_pages):
            page_text = reader.pages[i].extract_text()
            if page_text:
                texts.append(page_text.strip())
        return "\n\n".join(texts) if texts else None
    except Exception as e:
        print(f"Failed to extract from {pdf_path.name}: {e}")
        return None

def guess_pdf_metadata(llm: Anthropic, prompt_text: str) -> Optional[Dict[str, str]]:
    """
    Big-picture: Submit extracted text to Claude, requesting structured JSON metadata.
    Inputs: llm (Anthropic instance), prompt_text (string from PDF first page)
    Outputs: Dict with keys 'author', 'title', 'pubdate', or None on error/unreliability.
    Role: Central "brain" of the pipeline, abstracting LLM operations and filtering unreliable suggestions.
    """
    SYSTEM_PROMPT = (
        "You are a librarian interested in the organization of knowledge. "
        "You assist in renaming digital files to build a perfect library. "
        "Only respond in JSON with fields: author, title, pubdate as CamelCase strings. "
        "If unsure, print 'Various' for author or 'Unknown' for title. pubdate is four-digit year."
    )
    USER_MSG = (
        f"Given the following text, guess probable author (with a preference for institutions over individuals), title, and publication year. "
        f"Format like: OrgA & OrgB & Jane Smith - The Document Title- Subtitles (2023)."
        f"Strictly JSON: {{'author':'', 'title':'', 'pubdate':''}}. "
        f"----\n{prompt_text}\n----"
    )
    try:
        response = llm.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": USER_MSG,
            }]
        )
        content = response.content[0].text
        # Extract JSON from any codeblock wrapper
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
        content = match.group(1) if match else content
        # Remove leading/trailing text outside braces if present
        # Clean up common LLM output issues for JSON parsing
        # 1. Remove code block markers
        content = re.sub(r'^```(?:json)?', '', content.strip(), flags=re.IGNORECASE).strip()
        content = re.sub(r'```$', '', content.strip()).strip()
        # 2. Extract JSON object from surrounding text
        content_match = re.search(r'\{.*\}', content, flags=re.DOTALL)
        if content_match:
            content = content_match.group(0)
        # 3. Replace single quotes with double quotes (only outside of already valid JSON)
        #    Only do this if double quotes are not already present for keys
        if '"' not in content:
            content = content.replace("'", '"')
        # 4. Remove trailing commas
        content = re.sub(r',\s*([}\]])', r'\1', content)
        try:
            guessed = json.loads(content)
        except Exception as e2:
            print(f"LLM error (after cleaning): {e2}\nRaw content: {content}")
            return None
        # Reliability check:
        if (
            guessed.get("author", "").strip().lower() in {"unknown", "various"}
            or guessed.get("title", "").strip().lower() == "unknown"
            or not guessed.get("title", "").strip()
        ):
            return None
        return guessed
    except (Exception, json.JSONDecodeError) as e:
        print(f"LLM error: {e}")
        return None

def sanitize_filename(raw: str, limit: int = 200) -> str:
    """
    Big-picture: Remove forbidden/special chars, normalize whitespace, and ensure candidate filename is safe & not too long.
    Inputs: raw - candidate filename string; limit - char limit.
    Outputs: cleaned, truncated string.
    Role: Prevents OS errors, improves human readability in filenames.
    """
    # Accept: letters, numbers, underscores, hyphens, spaces, (), &
    cleaned = re.sub(r"[^\w\s\(\)\-\&]", "", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:limit]

def make_destination_path(base_dir: Path, proposed: str) -> Path:
    """
    Big-picture: Given a directory and candidate filename, return a unique, absolute Path that doesn't overwrite existing files.
    Inputs: base_dir - directory Path; proposed - proposed filename (may/may not end with .pdf).
    Outputs: Path object.
    Role: File system collision avoidance, multi-run safety.
    """
    # Always end with .pdf
    base_name = proposed
    if not base_name.lower().endswith(".pdf"):
        base_name += ".pdf"
    candidate = base_dir / base_name
    counter = 1
    while candidate.exists():
        stem, ext = os.path.splitext(base_name)
        candidate = base_dir / f"{stem}_{counter}{ext}"
        counter += 1
    return candidate

def update_and_save_pdf_metadata(
    src_pdf: Path, dest_pdf: Path, author: str, title: str, date_str: str
) -> bool:
    """
    Big-picture: Copy PDF, update XMP/document metadata, save as dest_pdf.
    Inputs: src_pdf (source path), dest_pdf (target), author/title (metadata), date_str (year)
    Outputs: True if successful, False on failure.
    Role: Ensures both correct filename and internal PDF metadata for archival integrity.
    """
    try:
        reader = PdfReader(str(src_pdf))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        year_candidate = str(date_str)
        metadata = {
            "/Author": author,
            "/Title": title,
            "/CreationDate": f"D:{year_candidate}0101000000Z"
        }
        writer.add_metadata(metadata)
        # Write to a safe temp, then move
        temp_path = dest_pdf.parent / (dest_pdf.name + ".tmp")
        with open(temp_path, "wb") as out_f:
            writer.write(out_f)
        temp_path.replace(dest_pdf)
        return True
    except Exception as e:
        print(f"Error updating/writing PDF ({src_pdf.name}): {e}")
        return False

def process_single_pdf(
    pdf_path: Path, llm: Anthropic
) -> Optional[Path]:
    """
    Big-picture: For a single PDF, extract, AI-infer metadata, attempt rename and metadata write.
    Inputs: pdf_path - PDF file; llm - Anthropic instance
    Outputs: Path to the new PDF file on success (or original path if unchanged/fail)
    Role: Unit of work for workflow; enables granular testing and extension (e.g., dry-run mode).
    """
    extracted = extract_first_n_pages_text(pdf_path, n=10)
    if not extracted:
        print(f"Skipping {pdf_path.name}: no text found.")
        return pdf_path
    guessed = guess_pdf_metadata(llm, extracted)
    if not guessed:
        print(f"Skipping {pdf_path.name}: LLM metadata guess failed or unreliable.")
        return pdf_path
    candidate_name = f"{guessed['author']} - {guessed['title']} ({guessed['pubdate']})"
    clean_file = sanitize_filename(candidate_name)
    new_path = make_destination_path(pdf_path.parent, clean_file)
    # Attempt to update metadata and rename atomically
    if update_and_save_pdf_metadata(pdf_path, new_path, sanitize_filename(guessed['author']),
                                    sanitize_filename(guessed['title']), guessed['pubdate']):
        # (Safety: Optionally, backup original instead of deleting/moving)
        pdf_path.unlink(missing_ok=True)
        print(f"Renamed '{pdf_path.name}' → '{new_path.name}'")
        return new_path
    else:
        print(f"Failed to process '{pdf_path.name}': metadata/write error.")
        return pdf_path

def process_pdf_directory(directory: Path, llm: Anthropic):
    """
    Big-picture: For all PDFs in directory, apply process_single_pdf() in sorted, recent-first order.
    Inputs: directory (Path), llm (Anthropic instance)
    Outputs: None (prints progress).
    Role: Batch driver for workflow; entrypoint for CLI, scripting, or extension.
    """
    pdfs = sorted((f for f in directory.iterdir() if f.suffix.lower() == ".pdf" and f.is_file()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for pdf_path in pdfs:
        process_single_pdf(pdf_path, llm)
    print("Finished processing all PDFs!")

def main():
    """
    Big-picture:
    1. Prompt for input directory (unless provided directly).
    2. Initialize Anthropic LLM instance securely.
    3. Pass directory contents to process_pdf_directory().
    4. Provide summary report of operation.
    Rationale: End-to-end orchestration; makes interactive, command-line, or script use all possible.
    """
    dir_arg = ""  # Hard-code here for automatic testing, leave blank to prompt.
    # Could accept sys.argv[1] for CLI extension.
    directory = Path(dir_arg) if dir_arg else Path(input("Enter directory with PDFs: ")).expanduser().resolve()
    if not directory.is_dir():
        print(f"Invalid directory: {directory}")
        return
    llm = load_llm()
    process_pdf_directory(directory, llm)

if __name__ == "__main__":
    main()
