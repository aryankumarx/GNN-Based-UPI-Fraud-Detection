import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

reports_remaining = [
    {
        "week": "Week 11",
        "title": "Adversarial Stress Testing & Edge Cases",
        "done": [
            "Conducted adversarial stress testing against Sybil attacks and rapid device ID spoofing.",
            "Verified that Rule Engine catches borderline cases where GNN confidence dips below 0.50."
        ],
        "next": [
            "Integrate GNNExplainer for subgraph visualization and nodal feature importance."
        ],
        "comments": "Adversarial evaluation confirms robustness against evasion attempts by fraud syndicates."
    },
    {
        "week": "Week 12",
        "title": "Explainable AI (XAI) & Interpretability",
        "done": [
            "Implemented GNNExplainer integration to extract 2-hop topological attribution subgraphs.",
            "Mapped model decisions to human-readable compliance tags for banking and RBI audit trails."
        ],
        "next": [
            "Execute multi-threaded concurrency and high-throughput stress testing."
        ],
        "comments": "XAI features provide strong regulatory compliance for financial decision audits."
    },
    {
        "week": "Week 13",
        "title": "High Concurrency & Load Testing",
        "done": [
            "Benchmarked system under simulated high load of 1,000 transactions/second on multi-core CPU.",
            "Maintained median P50 latency of 5.8 ms with zero dropped transactions or memory leaks."
        ],
        "next": [
            "Compile comprehensive project thesis, IEEE format research paper, and user guide."
        ],
        "comments": "Throughput and memory stability meet commercial payment processing criteria."
    },
    {
        "week": "Week 14",
        "title": "Research Paper Drafting & Documentation",
        "done": [
            "Drafted research paper titled 'Real-Time UPI Fraud Detection Using Inductive GraphSAGE'.",
            "Compiled complete technical thesis chapters covering methodology, loss formulas, and benchmarks."
        ],
        "next": [
            "Conduct mock viva presentations and practice interactive UI demo workflows."
        ],
        "comments": "Documentation and research paper draft are well-structured and technically rigorous."
    },
    {
        "week": "Week 15",
        "title": "Mock Viva Defense & Final Submission",
        "done": [
            "Rehearsed live examiner demo covering Auto-Stream, clean transactions, and mule ring attack.",
            "Finalized project deliverables, code repository archiving, and supervisor sign-offs."
        ],
        "next": [
            "Appear for final external project viva evaluation and committee defense."
        ],
        "comments": "Excellent execution across all engineering milestones. Fully approved for final examination."
    }
]

def set_cell_border(cell, color="000000", sz="12", val="single"):
    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:left w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:right w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)

def generate_docx(filename):
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    for i, r in enumerate(reports_remaining):
        p_head = doc.add_paragraph()
        run_week = p_head.add_run(f"Project Progress Report — {r['week']}: {r['title']}")
        run_week.bold = True
        run_week.font.size = Pt(13)
        run_week.font.color.rgb = RGBColor(14, 15, 12)
        p_head.paragraph_format.space_after = Pt(10)

        # Box 1
        t1 = doc.add_table(rows=1, cols=1)
        t1.alignment = WD_TABLE_ALIGNMENT.CENTER
        t1.autofit = False
        t1.columns[0].width = Inches(6.8)
        c1 = t1.cell(0, 0)
        set_cell_border(c1)
        
        p1 = c1.paragraphs[0]
        r1_title = p1.add_run("Work done in current Week:\n")
        r1_title.bold = True
        r1_title.font.size = Pt(11)
        
        for item in r["done"]:
            p_item = c1.add_paragraph()
            p_item.paragraph_format.left_indent = Inches(0.2)
            p_item.paragraph_format.space_after = Pt(2)
            r_bullet = p_item.add_run(f"• {item}")
            r_bullet.font.size = Pt(10.5)

        doc.add_paragraph().paragraph_format.space_after = Pt(4)

        # Box 2
        t2 = doc.add_table(rows=1, cols=1)
        t2.alignment = WD_TABLE_ALIGNMENT.CENTER
        t2.autofit = False
        t2.columns[0].width = Inches(6.8)
        c2 = t2.cell(0, 0)
        set_cell_border(c2)

        p2 = c2.paragraphs[0]
        r2_title = p2.add_run("Work to be done in next Week :\n")
        r2_title.bold = True
        r2_title.font.size = Pt(11)

        for item in r["next"]:
            p_item2 = c2.add_paragraph()
            p_item2.paragraph_format.left_indent = Inches(0.2)
            p_item2.paragraph_format.space_after = Pt(2)
            r_bullet2 = p_item2.add_run(f"• {item}")
            r_bullet2.font.size = Pt(10.5)

        doc.add_paragraph().paragraph_format.space_after = Pt(4)

        # Box 3
        t3 = doc.add_table(rows=1, cols=1)
        t3.alignment = WD_TABLE_ALIGNMENT.CENTER
        t3.autofit = False
        t3.columns[0].width = Inches(6.8)
        c3 = t3.cell(0, 0)
        set_cell_border(c3)

        p3 = c3.paragraphs[0]
        r3_title = p3.add_run("Supervisor Comments:\n")
        r3_title.bold = True
        r3_title.font.size = Pt(11)

        p_comm = c3.add_paragraph()
        p_comm.paragraph_format.left_indent = Inches(0.2)
        p_comm.paragraph_format.space_after = Pt(16)
        r_comm = p_comm.add_run(r["comments"])
        r_comm.font.size = Pt(10.5)
        r_comm.italic = True

        p_sig = c3.add_paragraph()
        p_sig.paragraph_format.space_after = Pt(4)
        r_sig = p_sig.add_run("Supervisor Signature/Date: .....................................................................")
        r_sig.bold = True
        r_sig.font.size = Pt(9.5)
        r_sig.font.color.rgb = RGBColor(80, 80, 80)

        if i < len(reports_remaining) - 1:
            doc.add_page_break()

    doc.save(filename)
    print(f"Word document saved to: {filename}")

def generate_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'RepTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=13, leading=16,
        textColor=colors.HexColor('#0e0f0c'), spaceAfter=12
    )

    box_title_style = ParagraphStyle(
        'BoxTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=11, leading=15,
        textColor=colors.black, spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'BulletStyle', parent=styles['Normal'],
        fontName='Helvetica', fontSize=10, leading=14,
        textColor=colors.HexColor('#222222'), leftIndent=12, spaceAfter=4
    )

    comment_style = ParagraphStyle(
        'CommentStyle', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=10, leading=14,
        textColor=colors.HexColor('#2d3748'), leftIndent=12, spaceAfter=18
    )

    sig_style = ParagraphStyle(
        'SigStyle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=9, leading=12,
        textColor=colors.HexColor('#4a5568')
    )

    elements = []

    for i, r in enumerate(reports_remaining):
        elements.append(Paragraph(f"Project Progress Report — {r['week']}: {r['title']}", title_style))

        # Box 1
        b1_content = [Paragraph("<b>Work done in current Week:</b>", box_title_style)]
        for item in r["done"]:
            b1_content.append(Paragraph(f"• {item}", bullet_style))
        b1_table = Table([[b1_content]], colWidths=[530])
        b1_table.setStyle(TableStyle([
            ('BOX', (0,0), (-1,-1), 1.2, colors.black),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
            ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ]))
        elements.append(b1_table)
        elements.append(Spacer(1, 14))

        # Box 2
        b2_content = [Paragraph("<b>Work to be done in next Week :</b>", box_title_style)]
        for item in r["next"]:
            b2_content.append(Paragraph(f"• {item}", bullet_style))
        b2_table = Table([[b2_content]], colWidths=[530])
        b2_table.setStyle(TableStyle([
            ('BOX', (0,0), (-1,-1), 1.2, colors.black),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
            ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ]))
        elements.append(b2_table)
        elements.append(Spacer(1, 14))

        # Box 3
        b3_content = [
            Paragraph("<b>Supervisor Comments:</b>", box_title_style),
            Paragraph(r["comments"], comment_style),
            Paragraph("<b>Supervisor Signature/Date:</b> .....................................................................................", sig_style)
        ]
        b3_table = Table([[b3_content]], colWidths=[530])
        b3_table.setStyle(TableStyle([
            ('BOX', (0,0), (-1,-1), 1.2, colors.black),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
            ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ]))
        elements.append(b3_table)

        if i < len(reports_remaining) - 1:
            elements.append(PageBreak())

    doc.build(elements)
    print(f"PDF document saved to: {filename}")

if __name__ == "__main__":
    docx_out = "College_Project_Progress_Report_Remaining_5_Pages.docx"
    pdf_out = "College_Project_Progress_Report_Remaining_5_Pages.pdf"
    generate_docx(docx_out)
    generate_pdf(pdf_out)

