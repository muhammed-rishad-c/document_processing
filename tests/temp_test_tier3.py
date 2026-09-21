from doc_processor.app.service import (
    extract_text_from_file,
    extract_document_structure,
    needs_tier3_llm_fallback,
)
from doc_processor.app import service as service_module


# Point this at any real PDF you want to test. No generation step, no
# overwriting anything -- this just reads the file you already have.
REAL_PDF_PATH = r"C:\Users\Rishad R\Downloads\Fables of the Whispering Woods.pdf"


def main():
    with open(REAL_PDF_PATH, "rb") as f:
        file_bytes = f.read()

    filename = REAL_PDF_PATH

    # ---- Step 1: same text extraction the real upload endpoint uses ----
    extracted_text, file_type = extract_text_from_file(file_bytes, filename)
    print(f"[step 1] extracted_text length={len(extracted_text)} chars, file_type={file_type}")

    # ---- Step 2: Tier 1 (embedded TOC) only ----
    structure = extract_document_structure(file_bytes, filename)
    print("\n[step 2] Tier 1 result:")
    print(f"    source        = {structure['source']}")
    print(f"    page_count    = {structure['page_count']}")
    print(f"    chapter_count = {structure['chapter_count']}")
    for c in structure["chapters"]:
        print(f"        - {c['title']}  (page={c['page']})")

    # ---- Step 3: would the real app schedule Tier 3 here? ----
    needs_tier3 = needs_tier3_llm_fallback(structure)
    print(f"\n[step 3] needs_tier3_llm_fallback = {needs_tier3}")

    if not needs_tier3:
        print(
            "\n-> Tier 1 found an embedded TOC in this real file, so in the "
            "real app Tier 3 would never run for this document. That's the "
            "correct/expected outcome for a PDF that already has a proper "
            "table of contents."
        )
        return

    print("\n[step 4] Tier 1 found nothing -> calling Tier 3 (hits your real LLM)...")
    tier3_result = service_module._extract_tier3_llm_structure(
        extracted_text, structure["page_count"]
    )

    print("\n[step 4] Tier 3 result:")
    print(f"    source        = {tier3_result['source']}")
    print(f"    page_count    = {tier3_result['page_count']}")
    print(f"    chapter_count = {tier3_result['chapter_count']}")
    for c in tier3_result["chapters"]:
        print(f"        - {c['title']}")

    if tier3_result["source"] == "llm_inferred" and tier3_result["chapter_count"] > 0:
        print("\n✅ Tier 3 ran successfully and returned chapters.")
        print(
            "   Check the list above: continuation markers should be merged "
            "into their parent chapter, cover-page decoration should be "
            "excluded, and back-matter/bonus sections should be excluded too."
        )
    elif tier3_result["source"] == "none":
        print(
            "\n⚠️  Tier 3 ran but returned no chapters. Check the "
            "'[_extract_tier3_llm_structure] failed: ...' print above/in your logs."
        )


if __name__ == "__main__":
    main()