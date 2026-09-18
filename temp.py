from doc_processor.app.service import extract_document_structure
import json

with open(r"C:\Users\Rishad R\Downloads\Pip_the_Fox_Story.pdf", "rb") as f:
    file_bytes = f.read()

result = extract_document_structure(file_bytes, "some_no_toc_book.pdf")
print(json.dumps(result, indent=2))
print("\nsource:", result["source"])
print("chapter_count:", result["chapter_count"])