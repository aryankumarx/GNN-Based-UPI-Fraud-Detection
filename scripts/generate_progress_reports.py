import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

reports = [
    {
        "week": "Week 1",
        "title": "Problem Definition & Literature Survey",
        "done": [
            "Researched UPI fraud vectors focusing on mule accounts, phishing, and rapid layering.",
            "Finalized problem statement, literature survey, and project requirement specifications."
        ],
        "next": [
            "Study Graph Neural Network algorithms (GCN vs GraphSAGE) and design entity schema."
        ],
        "comments": "Good literature review. Problem statement clearly identifies relational graph advantages."
    },
    {
        "week": "Week 2",
        "title": "Synthetic Data Generation",
        "done": [
            "Developed synthetic UPI transaction generator (100,000 transactions across 5,000 users).",
            "Injected 5 realistic fraud patterns (mule rings, nocturnal transfers, velocity bursts)."
        ],
        "next": [
            "Perform data preprocessing, cyclic time encoding, and feature normalization."
        ],
        "comments": "Realistic data distribution. Ensure 1.5% fraud class imbalance is maintained."
    },
    {
        "week": "Week 3",
        "title": "Feature Engineering & Graph Construction",
        "done": [
            "Built interaction graph connecting transactions via shared Device IDs and proxy IPs.",
            "Applied degree capping on high-volume merchant nodes to prevent over-smoothing."
        ],
        "next": [
            "Split graph into train, validation, and test masks using temporal ordering."
        ],
        "comments": "Graph construction is sound. Temporal ordering prevents data leakage."
    },
    {
        "week": "Week 4",
        "title": "GraphSAGE Architecture Design",
        "done": [
            "Implemented 2-layer Inductive GraphSAGE model in PyTorch Geometric.",
            "Constructed baseline models (XGBoost and 3-layer MLP) for empirical benchmarking."
        ],
        "next": [
            "Formulate custom loss function to address severe 1.5% class imbalance."
        ],
        "comments": "Inductive GraphSAGE is appropriate for dynamic real-time user onboarding."
    },
    {
        "week": "Week 5",
        "title": "Loss Function (Sigmoid Focal Loss)",
        "done": [
            "Implemented Sigmoid Focal Loss (alpha=0.85, gamma=2.0) to penalize easy benign samples.",
            "Configured Adam optimizer (lr=0.003) and ReduceLROnPlateau learning rate scheduler."
        ],
        "next": [
            "Train GraphSAGE and perform validation threshold tuning on PR-curve."
        ],
        "comments": "Mathematical formulation of Focal Loss is sound. Monitor validation loss curves."
    },
    {
        "week": "Week 6",
        "title": "Model Training & Baseline Benchmarking",
        "done": [
            "Trained GraphSAGE over 35 epochs; achieved 99.62% precision with 0.884 PR-AUC.",
            "Evaluated test set: only 1 false positive out of 14,727 transactions (beating XGBoost by 14%)."
        ],
        "next": [
            "Build deterministic rule engine for cold-start and instant velocity checks."
        ],
        "comments": "Excellent benchmark results. High precision preserves clean customer experience."
    },
    {
        "week": "Week 7",
        "title": "70/30 Hybrid Decision Engine",
        "done": [
            "Formulated Hybrid Engine: Final Risk = 0.70 * GNN Score + 0.30 * Rule Score.",
            "Added Explainable AI (XAI) audit tagging for banking and RBI regulatory compliance."
        ],
        "next": [
            "Build production asynchronous FastAPI backend microservice."
        ],
        "comments": "Hybrid architecture effectively solves cold-start and regulatory explainability."
    },
    {
        "week": "Week 8",
        "title": "FastAPI Backend & Latency Benchmarks",
        "done": [
            "Developed high-throughput FastAPI service with /predict, /stats, and /health endpoints.",
            "Profiled single-sample latency: achieved 6.4 ms (well within NPCI's 50 ms SLA)."
        ],
        "next": [
            "Build interactive frontend dashboard with live network topology visualization."
        ],
        "comments": "Sub-10ms latency demonstrates production-grade real-time capability."
    },
    {
        "week": "Week 9",
        "title": "Wise-Themed Dashboard & Vis.js Topology",
        "done": [
            "Designed frontend in Wise Design System with centered Vis.js ForceAtlas2 network graph.",
            "Fixed graph spring jittering using 0.92 damping, minVelocity cutoff, and radial anchors."
        ],
        "next": [
            "Deploy multi-service project on Vercel and execute test suites."
        ],
        "comments": "Visual interface is exceptionally clean. Mule ring simulation works smoothly."
    },
    {
        "week": "Week 10",
        "title": "Cloud Deployment & Final Verification",
        "done": [
            "Configured Vercel multi-service deployment with decoupled edge frontend and serverless API.",
            "Verified 10/10 automated pytest unit suites and completed thesis viva preparation."
        ],
        "next": [
            "Final academic presentation, live demonstration, and project thesis submission."
        ],
        "comments": "All project milestones completed successfully. Approved for final evaluation."
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
    
    # Page margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    for i, r in enumerate(reports):
        # Header / Title
        p_head = doc.add_paragraph()
        run_week = p_head.add_run(f"Project Progress Report — {r['week']}: {r['title']}")
        run_week.bold = True
        run_week.font.size = Pt(13)
        run_week.font.color.rgb = RGBColor(14, 15, 12)
        p_head.paragraph_format.space_after = Pt(10)

        # Box 1: Work done in current Week:
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
        r1_title.font.color.rgb = RGBColor(0, 0, 0)
        
        for item in r["done"]:
            p_item = c1.add_paragraph()
            p_item.paragraph_format.left_indent = Inches(0.2)
            p_item.paragraph_format.space_after = Pt(2)
            r_bullet = p_item.add_run(f"• {item}")
            r_bullet.font.size = Pt(10.5)

        # Spacing
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

        # Box 2: Work to be done in next Week :
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
        r2_title.font.color.rgb = RGBColor(0, 0, 0)

        for item in r["next"]:
            p_item2 = c2.add_paragraph()
            p_item2.paragraph_format.left_indent = Inches(0.2)
            p_item2.paragraph_format.space_after = Pt(2)
            r_bullet2 = p_item2.add_run(f"• {item}")
            r_bullet2.font.size = Pt(10.5)

        # Spacing
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

        # Box 3: Supervisor Comments:
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
        r3_title.font.color.rgb = RGBColor(0, 0, 0)

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

        # Add page break except for last item
        if i < len(reports) - 1:
            doc.add_page_break()

    doc.save(filename)
    print(f"Word document saved successfully to: {filename}")

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
        'RepTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0e0f0c'),
        spaceAfter=12
    )

    box_title_style = ParagraphStyle(
        'BoxTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.black,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#222222'),
        leftIndent=12,
        spaceAfter=4
    )

    comment_style = ParagraphStyle(
        'CommentStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2d3748'),
        leftIndent=12,
        spaceAfter=18
    )

    sig_style = ParagraphStyle(
        'SigStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#4a5568')
    )

    elements = []

    for i, r in enumerate(reports):
        # Page title
        elements.append(Paragraph(f"Project Progress Report — {r['week']}: {r['title']}", title_style))

        # Box 1: Work done
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

        # Box 2: Work to be done
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

        # Box 3: Supervisor comments
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

        if i < len(reports) - 1:
            elements.append(PageBreak())

    doc.build(elements)
    print(f"PDF document saved successfully to: {filename}")

if __name__ == "__main__":
    docx_out = "College_Project_Progress_Report_10_Pages.docx"
    pdf_out = "College_Project_Progress_Report_10_Pages.pdf"
    generate_docx(docx_out)
    generate_pdf(pdf_out)

