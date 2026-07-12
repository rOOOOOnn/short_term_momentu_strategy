from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs/reports/premarket_strategy_summary/premarket_strategy_research_summary_zh.md"
DEFAULT_OUTPUT = ROOT / "docs/reports/premarket_strategy_summary/premarket_strategy_research_summary_zh.pdf"

IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")


def register_fonts() -> tuple[str, str]:
    chinese_font = "STSong-Light"
    pdfmetrics.registerFont(UnicodeCIDFont(chinese_font))
    return chinese_font, "Courier"


def make_styles(base_font: str, mono_font: str) -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName=base_font,
            fontSize=22,
            leading=30,
            textColor=colors.HexColor("#172033"),
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "Heading2CN",
            parent=styles["Heading2"],
            fontName=base_font,
            fontSize=15,
            leading=21,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=14,
            spaceAfter=7,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=9.8,
            leading=15,
            textColor=colors.HexColor("#222222"),
            alignment=TA_LEFT,
            spaceAfter=7,
        ),
        "bullet": ParagraphStyle(
            "BulletCN",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=9.5,
            leading=14,
            leftIndent=16,
            firstLineIndent=-10,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "CaptionCN",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=11,
            textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=8,
        ),
        "table": ParagraphStyle(
            "TableCN",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=6.2,
            leading=8.2,
            wordWrap="CJK",
        ),
        "mono": ParagraphStyle(
            "Mono",
            parent=styles["BodyText"],
            fontName=mono_font,
            fontSize=7,
            leading=9,
        ),
    }


def inline_markup(text: str, mono_font: str) -> str:
    escaped = html.escape(html.unescape(text))

    def replace_code(match: re.Match[str]) -> str:
        return f'<font name="{mono_font}" size="8">{html.escape(match.group(1))}</font>'

    return INLINE_CODE_RE.sub(replace_code, escaped)


def is_separator_row(row: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in row)


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        raw = lines[i].strip().strip("|")
        row = [cell.strip() for cell in raw.split("|")]
        if row and not is_separator_row(row):
            rows.append(row)
        i += 1
    return rows, i


def table_flowable(rows: list[list[str]], styles: dict[str, ParagraphStyle], available_width: float, mono_font: str) -> Table:
    col_count = max(len(row) for row in rows)
    normalized = [row + [""] * (col_count - len(row)) for row in rows]
    data = [
        [Paragraph(inline_markup(cell, mono_font), styles["table"]) for cell in row]
        for row in normalized
    ]
    col_widths = [available_width / col_count] * col_count
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("FONTSIZE", (0, 0), (-1, 0), 6.2),
                ("FONTSIZE", (0, 1), (-1, -1), 6),
                ("LEADING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd7e3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def image_flowables(
    markdown_dir: Path,
    image_path: str,
    alt: str,
    styles: dict[str, ParagraphStyle],
    available_width: float,
) -> list:
    path = (markdown_dir / image_path).resolve()
    if not path.exists():
        return [Paragraph(f"Missing image: {html.escape(image_path)}", styles["body"])]

    img = Image(str(path))
    max_width = available_width
    max_height = 110 * mm
    scale = min(max_width / img.imageWidth, max_height / img.imageHeight, 1.0)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    return [
        Spacer(1, 5),
        img,
        Paragraph(inline_markup(alt, "Courier"), styles["caption"]),
    ]


def paragraph_block(lines: list[str], start: int) -> tuple[str, int]:
    parts: list[str] = []
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            break
        if stripped.startswith(("# ", "## ", "- ", "|", "![")) or re.match(r"\d+\. ", stripped):
            break
        parts.append(stripped)
        i += 1
    return " ".join(parts), i


def build_story(markdown_path: Path, styles: dict[str, ParagraphStyle], page_width: float, mono_font: str) -> list:
    text = markdown_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    story: list = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        image_match = IMAGE_RE.fullmatch(line)
        if image_match:
            story.extend(
                image_flowables(
                    markdown_path.parent,
                    image_match.group("path"),
                    image_match.group("alt"),
                    styles,
                    page_width,
                )
            )
            i += 1
            continue

        if line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:].strip(), mono_font), styles["title"]))
            story.append(Spacer(1, 5))
            i += 1
            continue

        if line.startswith("## "):
            if story:
                story.append(Spacer(1, 4))
            story.append(Paragraph(inline_markup(line[3:].strip(), mono_font), styles["h2"]))
            i += 1
            continue

        if line.startswith("- "):
            story.append(Paragraph(inline_markup(line[2:].strip(), mono_font), styles["bullet"], bulletText="-"))
            i += 1
            continue

        number_match = re.match(r"(\d+)\.\s+(.*)", line)
        if number_match:
            story.append(
                Paragraph(
                    inline_markup(number_match.group(2), mono_font),
                    styles["bullet"],
                    bulletText=f"{number_match.group(1)}.",
                )
            )
            i += 1
            continue

        if line.startswith("|"):
            rows, i = parse_table(lines, i)
            if rows:
                story.append(Spacer(1, 4))
                story.append(table_flowable(rows, styles, page_width, mono_font))
                story.append(Spacer(1, 8))
            continue

        para, i = paragraph_block(lines, i)
        if para:
            story.append(Paragraph(inline_markup(para, mono_font), styles["body"]))
        else:
            i += 1

    return story


def draw_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("STSong-Light", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(doc.leftMargin, 11 * mm, "盘前动量策略研究总结")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 11 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def export_pdf(markdown_path: Path, output_path: Path) -> None:
    base_font, mono_font = register_fonts()
    styles = make_styles(base_font, mono_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pagesize = landscape(A4)
    left_margin = right_margin = 12 * mm
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=pagesize,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=13 * mm,
        bottomMargin=17 * mm,
        title="Premarket Strategy Research Summary",
        author="short_term_momentu_strategy",
    )
    available_width = pagesize[0] - left_margin - right_margin
    story = build_story(markdown_path, styles, available_width, mono_font)
    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the Chinese research summary Markdown report to PDF.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_pdf(args.input.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
