"""Generate a synthetic contract corpus for tests and the retrieval benchmark.

Everything here is invented. It imitates the layout problems of real
contract PDFs: running headers/footers, a table of contents, numbered
definitions, clauses that span pages, cross-references that wrap to the
start of a line ("10.2 above"), percentages at line starts, an appendix
table, and two documents that both have a clause 10.7.

    uv run python tests/fixtures/build_corpus.py <out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from fpdf import FPDF

FILLER = (
    "The Contractor shall carry out all Petroleum Operations diligently, in accordance with modern oilfield "
    "and petroleum industry practices, and in compliance with applicable laws, rules and regulations. "
)


class ContractPDF(FPDF):
    def __init__(self, header_text: str):
        super().__init__(format="A4")
        self.header_text = header_text
        self.set_auto_page_break(True, margin=22)
        self.set_margins(22, 22, 22)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, self.header_text, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, f"Page {self.page_no()} of {{nb}}", align="C")

    def title_block(self, *lines: str):
        self.set_font("Helvetica", "B", 15)
        for line in lines:
            self.multi_cell(0, 9, line, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def heading(self, text: str):
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")

    def para(self, text: str):
        self.set_font("Helvetica", "", 10.5)
        self.multi_cell(0, 5.6, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)


def _save(pdf: FPDF, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))


def kgd6_psc(out: Path) -> None:
    pdf = ContractPDF("KG-DWN-98/3 Production Sharing Contract - Confidential")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.title_block("PRODUCTION SHARING CONTRACT", "BLOCK KG-DWN-98/3")
    pdf.para(
        "This Contract is made on this 12th day of April, 2000 between THE PRESIDENT OF INDIA, acting through the "
        "Joint Secretary, Ministry of Petroleum and Natural Gas (hereinafter referred to as the Government), and "
        "RELIANCE INDUSTRIES LIMITED, a company incorporated in India (RIL), and NIKO RESOURCES LIMITED, a company "
        "incorporated in Canada (Niko), (hereinafter together referred to as the Contractor)."
    )
    pdf.para("WHEREAS the Government wishes to promote exploration of the deep water block known as KG-D6.")
    pdf.add_page()
    pdf.heading("TABLE OF CONTENTS")
    for line in [
        "ARTICLE 1 DEFINITIONS ........................ 3",
        "ARTICLE 2 PARTICIPATING INTEREST ............. 5",
        "ARTICLE 10 DISCOVERY, DEVELOPMENT AND PRODUCTION ..... 6",
        "ARTICLE 16 ROYALTY ............................ 9",
        "ARTICLE 21 RELINQUISHMENT ..................... 10",
        "ARTICLE 27 ASSIGNMENT OF PARTICIPATING INTEREST ...... 11",
        "APPENDIX C PROFIT PETROLEUM ................... 12",
        "ARTICLE 30 NOTICES ............................ 13",
    ]:
        pdf.para(line)
    pdf.add_page()
    pdf.heading("ARTICLE 1")
    pdf.heading("DEFINITIONS")
    pdf.para("1.1 In this Contract, unless the context otherwise requires, the following terms shall have the meanings set out below.")
    defs = [
        ("1.2", "Appraisal Programme", "means a programme carried out following a Discovery for the purpose of delineating the Reservoir to which the Discovery relates in terms of thickness and lateral extent and estimating the quantity of recoverable Petroleum therein."),
        ("1.3", "Commercial Discovery", "means a Discovery which has been declared to be commercial in accordance with Article 10.6."),
        ("1.4", "Contract Area", "means the area described in Appendix A, as reduced by relinquishment under Article 21."),
        ("1.5", "Discovery", "means the finding, during Petroleum Operations, of a deposit of Petroleum not previously known to have existed, which can be recovered at the surface in a flow measurable by conventional petroleum industry testing methods."),
        ("1.6", "Discovery Area", "means that part of the Contract Area about which, based on Discovery and the results obtained from a well or wells drilled in such part, the Contractor is of the opinion that Petroleum exists and is likely to be produced in commercial quantities."),
        ("1.7", "Development Area", "means that part of the Contract Area which encompasses one or more Commercial Discoveries and as delineated in the approved Development Plan."),
        ("1.8", "Investment Multiple", "means the ratio of accumulated net cash income to accumulated investment, computed in accordance with Appendix C."),
        ("1.9", "Petroleum", "means Crude Oil and/or Natural Gas existing in their natural condition."),
        ("1.10", "Reservoir", "means a naturally occurring discrete accumulation of Petroleum."),
    ]
    for num, term, text in defs:
        pdf.para(f'{num} "{term}" {text}')
    pdf.add_page()
    pdf.heading("ARTICLE 2 - PARTICIPATING INTEREST")
    pdf.para(
        "2.1 The Participating Interests of the Companies constituting the Contractor on the Effective Date shall be: "
        "Reliance Industries Limited - ninety per cent (90%); Niko Resources Limited - ten per cent (10%)."
    )
    pdf.para("2.2 Any change in the Participating Interest shall be made only in accordance with Article 27.")
    pdf.add_page()
    pdf.heading("ARTICLE 10")
    pdf.heading("DISCOVERY, DEVELOPMENT AND PRODUCTION")
    pdf.para("10.1 If and when a Discovery is made, the Contractor shall promptly, and in any event within thirty (30) days, notify the Management Committee and the Government of the Discovery.")
    pdf.para(
        "10.2 Within sixty (60) days of the notification under Article 10.1, the Contractor shall submit to the Management "
        "Committee particulars of the Discovery, including its potential commercial interest. The obligations in Article\n"
        "10.2 above shall apply to every Discovery."
    )
    pdf.para("10.3 If the Contractor considers that the Discovery merits appraisal, it shall submit an Appraisal Programme within one hundred and eighty (180) days from the date of notification of the Discovery.")
    pdf.para("10.4 " + FILLER * 3)
    pdf.para("10.5 The royalty payable shall be calculated at the rate of\n12.5 per cent of the well-head value only where Article 16 so provides.")
    pdf.para("10.6 Declaration of Commerciality. Upon completion of the Appraisal Programme the Contractor shall, within ninety (90) days, declare whether the Discovery is a Commercial Discovery.")
    pdf.para(
        "10.7 Development Plan. Where a Discovery has been declared a Commercial Discovery, the Contractor shall, within "
        "two hundred (200) days from the date of such declaration, submit to the Management Committee a Development Plan "
        "for the Development Area, including the delineation of the Reservoir, production profiles, and estimated "
        "costs. " + FILLER * 6 +
        "(a) the Development Plan shall be approved by the Management Committee within one hundred and twenty (120) days of its submission; "
        "(b) where the Management Committee fails to approve it, the Contractor may refer the matter to the Government under Article 10.9."
    )
    pdf.para("10.8 " + FILLER * 2)
    pdf.para("10.9 Any dispute regarding the Development Plan shall be resolved under Article 33.")
    pdf.add_page()
    pdf.heading("ARTICLE 16")
    pdf.heading("ROYALTY")
    pdf.para("16.1 The Contractor shall pay royalty to the Government on the well-head value of Petroleum produced and saved from the Contract Area.")
    pdf.para("16.2 The rate of royalty for offshore areas shall be ten per cent (10%) of the well-head value of Natural Gas.")
    pdf.add_page()
    pdf.heading("ARTICLE 21")
    pdf.heading("RELINQUISHMENT")
    pdf.para("21.1 The Contractor shall relinquish twenty-five per cent (25%) of the original Contract Area at the end of Phase I.")
    pdf.para("21.5 Right of retention. The Contractor shall have the right to retain a Development Area for so long as Petroleum is produced in commercial quantities from it, subject to Article 10.7. If no Development Plan is submitted within the period in Article 10.7, the Contractor shall relinquish the Discovery Area.")
    pdf.add_page()
    pdf.heading("ARTICLE 27")
    pdf.heading("ASSIGNMENT OF PARTICIPATING INTEREST")
    pdf.para("27.1 Subject to the terms of this Article, a Company may assign, sell, transfer or otherwise convey all or part of its Participating Interest with the prior written consent of the Government, which consent shall not be unreasonably withheld.")
    pdf.para("27.2 No assignment shall be effective until the assignee has executed a deed of adherence and the Government has approved the assignment in writing.")
    pdf.add_page()
    pdf.heading("APPENDIX C")
    pdf.heading("PROFIT PETROLEUM SHARE")
    pdf.para("The share of Profit Petroleum between the Government and the Contractor shall be determined by the Investment Multiple (IM) as follows:")
    for row in [
        "IM less than 1.5: Government 10%, Contractor 90%",
        "IM 1.5 or more but less than 2.0: Government 16%, Contractor 84%",
        "IM 2.0 or more but less than 2.5: Government 28%, Contractor 72%",
        "IM 2.5 or more but less than 3.5: Government 40%, Contractor 60%",
        "IM 3.5 or more: Government 85%, Contractor 15%",
    ]:
        pdf.para(row)
    pdf.add_page()
    pdf.heading("ARTICLE 30 - NOTICES")
    pdf.para("30.1 All notices shall be in writing and delivered to the addresses set out in this Article.")
    _save(pdf, out / "PSC/Domestic/KG-DWN-98_3 PSC.pdf")


def nec_joa(out: Path) -> None:
    pdf = ContractPDF("NEC-OSN-97/2 Joint Operating Agreement")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.title_block("JOINT OPERATING AGREEMENT", "BLOCK NEC-OSN-97/2")
    pdf.para("This Joint Operating Agreement is made on 3rd day of March, 2001 between Reliance Industries Limited (Operator) and Niko Resources Limited (Non-Operator).")
    pdf.heading("ARTICLE 1 DEFINITIONS")
    pdf.para('1.1 "Discovery" shall have the meaning given in the Production Sharing Contract for Block NEC-OSN-97/2.')
    pdf.para('1.2 "Operating Committee" means the committee constituted under Article 5.')
    pdf.heading("ARTICLE 10 - SOLE RISK OPERATIONS")
    pdf.para("10.1 Any Party may propose a Sole Risk operation that has not been approved by the Operating Committee.")
    pdf.para("10.7 Sole Risk Development. A Party carrying out a Sole Risk Development shall bear one hundred per cent (100%) of its costs and shall be entitled to all production from it until it has recovered four hundred per cent (400%) of such costs.")
    pdf.heading("ARTICLE 12 - TRANSFER OF INTEREST")
    pdf.para("12.1 A Party may transfer its Participating Interest only after giving the other Parties a right of first refusal for a period of thirty (30) days.")
    _save(pdf, out / "JOA/New JOA/NEC-OSN-97-2 JOA.pdf")


def amendment(out: Path) -> None:
    pdf = ContractPDF("Amendment No. 1 to the KG-DWN-98/3 PSC")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.title_block("AMENDMENT NO. 1 TO THE PRODUCTION SHARING CONTRACT", "BLOCK KG-DWN-98/3")
    pdf.para("This Amendment is made on 14th day of June, 2018 between the Government of India and the Contractor under the Production Sharing Contract for Block KG-DWN-98/3 dated 12 April 2000.")
    pdf.para("1. Article 16.2 of the Contract is deleted and replaced with the following: \"16.2 The rate of royalty for offshore areas shall be twelve and one half per cent (12.5%) of the well-head value of Natural Gas.\"")
    pdf.para("2. This Amendment shall take effect from 1 August 2018. All other terms of the Contract remain unchanged.")
    _save(pdf, out / "PSC/Domestic/KG-D6 PSC Amendment No 1.pdf")


def dgh_letter(out: Path) -> None:
    pdf = ContractPDF("Directorate General of Hydrocarbons")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.title_block("DIRECTORATE GENERAL OF HYDROCARBONS", "APPROVAL OF ASSIGNMENT")
    pdf.para("Ref: DGH/KG-D6/ASSIGN/2018 Date: 20 July 2018")
    pdf.para("To: Reliance Industries Limited")
    pdf.para("Subject: Assignment of thirty per cent (30%) Participating Interest in Block KG-DWN-98/3 from Horizon Energy to NBP Resources.")
    pdf.para("The Ministry of Petroleum and Natural Gas (MoPNG) has approved the assignment of thirty per cent (30%) Participating Interest from Horizon Energy to NBP Resources with effect from 1 August 2018, subject to execution of the deed of adherence under Article 27.2 of the Production Sharing Contract.")
    _save(pdf, out / "DGH/DGH approval assignment 2018.pdf")


def minutes(out: Path) -> None:
    pdf = ContractPDF("KG-D6 Management Committee")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.title_block("MINUTES OF THE 42ND MANAGEMENT COMMITTEE MEETING", "BLOCK KG-DWN-98/3")
    pdf.para("Date: 18 June 2019. Venue: New Delhi.")
    pdf.para("Item 3: Royalty. The Management Committee noted that royalty continues to be paid at ten per cent (10%) of the well-head value pending clarification from the Government on Amendment No. 1.")
    pdf.para("Item 4: Work Programme and Budget. The Management Committee approved the Work Programme and Budget for 2019-20.")
    _save(pdf, out / "sites/Regulatory/KGD6/MINUTESnRESOLUTIONS/OCR/Document Center/MCM minutes 18 June 2019.pdf")


def build(out: Path) -> None:
    kgd6_psc(out)
    nec_joa(out)
    amendment(out)
    dgh_letter(out)
    minutes(out)


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "data/corpus"))
