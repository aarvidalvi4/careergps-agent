"""Generate the sample resume PDFs in tests/samples/.

Run: .venv/Scripts/python.exe -m tests.make_samples

Student names and emails are fictional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"


@dataclass
class Resume:
    filename: str
    name: str
    contact: str
    education: list[str]
    skills: list[str]
    projects: list[tuple[str, str]]  # (title, description)
    experience: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)


RESUMES = [
    Resume(
        filename="strong_data_analyst.pdf",
        name="Priya Deshmukh",
        contact="priya.deshmukh@example.com | Nagpur, Maharashtra",
        education=["B.Sc. Data Science, Shri Ramdeobaba College, Nagpur (RTMNU), 2022-2025, CGPA 8.6"],
        skills=["Python", "Pandas", "NumPy", "SQL (PostgreSQL)", "Advanced Excel", "Power BI", "Statistics", "Git"],
        projects=[
            (
                "APMC Mandi Price Dashboard (Power BI)",
                "Built an interactive Power BI dashboard on APMC mandi commodity prices from data.gov.in, "
                "tracking price trends and arrivals across Maharashtra markets.",
            ),
            (
                "Food-Delivery Orders SQL Case Study (PostgreSQL)",
                "Analysed food-delivery orders using window functions and CTEs to find repeat customers, "
                "peak hours and restaurant-level revenue.",
            ),
            (
                "Hypothesis Testing on Student Exam Scores (Python)",
                "Applied t-tests and chi-square tests in Python to check whether coaching and "
                "attendance affect exam scores; summarised findings in a short report.",
            ),
        ],
        experience=[
            "Data Intern, Prayas Foundation (NGO), Nagpur, Jan-Mar 2025 (3 months): built monthly Excel reports "
            "with pivot tables and charts on beneficiary programmes for the operations team."
        ],
    ),
    Resume(
        filename="medium_data_analyst.pdf",
        name="Rohan Patil",
        contact="rohan.patil@example.com | Kolhapur, Maharashtra",
        education=["B.Com, Shivaji University, Kolhapur, 2022-2025"],
        skills=["MS Excel", "Python (basics)", "Communication", "Presentation"],
        projects=[
            (
                "College Fest Budget Tracker (Excel)",
                "Created an Excel workbook to track sponsorships and expenses for the college fest, "
                "with pivot tables summarising spend by committee.",
            )
        ],
        certifications=["NPTEL: The Joy of Computing using Python"],
    ),
    Resume(
        filename="weak_ml_engineer.pdf",
        name="Ankit Yadav",
        contact="ankit.yadav@example.com | Jhansi, Uttar Pradesh",
        education=["BCA, Bundelkhand University, Jhansi, 2022-2025"],
        skills=["C", "HTML", "MS Office"],
        projects=[
            (
                "Library Management System (C)",
                "A console program in C to add, issue and return books, storing records in files.",
            )
        ],
    ),
]


def build(resume: Resume, out_dir: Path) -> Path:
    styles = getSampleStyleSheet()
    name_style = ParagraphStyle("Name", parent=styles["Title"], fontSize=18, spaceAfter=2)
    contact_style = ParagraphStyle("Contact", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10)
    heading = ParagraphStyle("Heading", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12, spaceBefore=8, spaceAfter=3)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=13)

    story = [Paragraph(resume.name, name_style), Paragraph(resume.contact, contact_style), Spacer(1, 4 * mm)]

    def section(title: str, lines: list[str]) -> None:
        if not lines:
            return
        story.append(Paragraph(title, heading))
        story.extend(Paragraph(line, body) for line in lines)

    section("Education", resume.education)
    section("Skills", [", ".join(resume.skills)])
    section("Projects", [f"<b>{title}</b>: {desc}" for title, desc in resume.projects])
    section("Experience", resume.experience)
    section("Certifications", resume.certifications)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / resume.filename
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm, bottomMargin=15 * mm)
    doc.build(story)
    return path


def main() -> None:
    for resume in RESUMES:
        print(f"wrote {build(resume, SAMPLES_DIR)}")


if __name__ == "__main__":
    main()
