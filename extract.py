import fitz

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

pdf_paths = ["Apple_Vision.pdf", "Apple_PER.pdf"]
pdf_texts = [extract_text_from_pdf(pdf_path) for pdf_path in pdf_paths]
print(pdf_texts)
