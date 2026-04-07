from rag_pipeline.loader import load_docs
from fpdf import FPDF

def test_loader_reads_pdfs(tmp_path):
    # Create a valid PDF file
    pdf_file = tmp_path / "sample.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Hello World", ln=True)
    pdf.output(str(pdf_file))

    docs = load_docs(str(tmp_path))
    assert docs[0]["id"] == "sample.pdf"
    assert "Hello World" in docs[0]["text"]
