import json
import argparse
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


def iter_block_items(parent):
    """Yield paragraphs and tables in order"""
    for child in parent.element.body.iterchildren():
        if child.tag.endswith('p'):
            yield Paragraph(child, parent)
        elif child.tag.endswith('tbl'):
            yield Table(child, parent)


def get_font_size(paragraph):
    sizes = []
    for run in paragraph.runs:
        if run.font.size:
            sizes.append(run.font.size.pt)
    return max(sizes) if sizes else None


def is_blue_heading(paragraph):
    size = get_font_size(paragraph)
    return size and size >= 14


def is_bold_subheading(paragraph):
    text = paragraph.text.strip()
    if not text:
        return False

    if len(text.split()) > 8:
        return False

    bold_runs = [r.bold for r in paragraph.runs if r.text.strip()]

    return bold_runs and all(bold_runs)


def table_to_text(table):
    rows = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def convert(docx_file, output_file):

    doc = Document(docx_file)

    dataset = []
    heading = None
    buffer = []

    for block in iter_block_items(doc):

        # paragraph
        if isinstance(block, Paragraph):

            text = block.text.strip()

            if not text:
                continue

            if is_blue_heading(block):

                if heading and buffer:
                    dataset.append({
                        "instruction": heading,
                        "output": "\n".join(buffer)
                    })

                heading = text
                buffer = []
                continue

            if is_bold_subheading(block):

                if heading and buffer:
                    dataset.append({
                        "instruction": heading,
                        "output": "\n".join(buffer)
                    })

                heading = text
                buffer = []
                continue

            buffer.append(text)

        # table
        elif isinstance(block, Table):

            table_text = table_to_text(block)
            buffer.append(table_text)

    # last section
    if heading and buffer:
        dataset.append({
            "instruction": heading,
            "output": "\n".join(buffer)
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Sections:", len(dataset))
    print("Saved:", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="dataset_tables.jsonl")

    args = parser.parse_args()

    convert(args.input, args.output)