import fitz
from pathlib import Path

def create_sample_pdf(filename: str, content: str):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), content)
    doc.save(filename)
    doc.close()
    print(f"Created {filename}")

if __name__ == "__main__":
    p1_text = """
    Document A: The Impact of Climate Change on Polar Bears.
    Polar bears (Ursus maritimus) are increasingly threatened by the loss of Arctic sea ice. 
    Studies show that by 2050, populations could decline by 30% [Ref-A1].
    The primary food source for polar bears is ringed seals.
    """
    
    p2_text = """
    Document B: Renewable Energy Trends 2024.
    Solar energy production has increased by 400% in the last decade.
    Wind power now accounts for 15% of global electricity generation.
    The cost of lithium-ion batteries has dropped by 80% since 2010.
    """
    
    create_sample_pdf("paper_a.pdf", p1_text)
    create_sample_pdf("paper_b.pdf", p2_text)
