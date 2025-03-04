#anthropic-pdf-renamer

This script does the following:

- takes a folder full of badly-named PDFs,
- extracts text from the first page,
- sends that text to Anthropic to guess the title, author and date of first publication
- and renames files with the results.

**Caution**

Please manage the security of your own data. Any loss of personal content due to errors in this script are your responsibility. Vet, test and modify the script to meet your needs.

**Known issues**

This script will not work on OCR image-based PDFs. The process will return Unknown - Unknown values for metadata, and documents could get rewritten. v0.5 preserves original filenames in that case, but do consider keeping backups of important data and checking total file counts if need be.
