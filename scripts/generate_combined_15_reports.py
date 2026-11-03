from generate_progress_reports import reports, generate_docx, generate_pdf
from generate_remaining_5_reports import reports_remaining
import generate_progress_reports as gr

if __name__ == "__main__":
    all_reports = reports + reports_remaining
    gr.reports = all_reports
    gr.generate_docx("College_Project_Progress_Report_All_15_Pages.docx")
    gr.generate_pdf("College_Project_Progress_Report_All_15_Pages.pdf")
    print("Combined 15 pages generated successfully!")

