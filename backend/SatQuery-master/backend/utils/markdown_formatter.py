"""
SatQuery Markdown & Table Formatting Utilities
==============================================
Provides:
1. Deterministic GFM table generation.
2. Robust markdown normalization and sanitization to fix malformed tables,
   concatenated header columns, and delimiter boundary errors.
"""

import re
from typing import Any, List, Optional


def format_markdown_table(
    headers: List[str],
    rows: List[List[Any]],
    alignments: Optional[List[str]] = None,
) -> str:
    """
    Generates a strictly compliant GitHub Flavored Markdown (GFM) table.
    
    Guarantees:
    - Preceding and trailing blank lines (\n\n) so parsers never merge with paragraphs.
    - Exactly one header row.
    - Exactly one separator row matching column count and alignment specs.
    - Every row has the exact same number of columns as the header.
    - No raw unescaped pipe characters inside cell content.
    - Clean whitespace padding around cell values.
    """
    if not headers:
        return ""

    num_cols = len(headers)
    clean_headers = [str(h).replace("|", "&#124;").strip() for h in headers]

    # Normalize alignments: 'left', 'right', 'center', or GFM strings ':---', '---:', ':---:'
    clean_align = []
    for i in range(num_cols):
        raw_a = (alignments[i] if alignments and i < len(alignments) else "left").lower().strip()
        if raw_a in ("right", "---:"):
            clean_align.append("---:")
        elif raw_a in ("center", ":---:"):
            clean_align.append(":---:")
        else:
            clean_align.append(":---")

    lines = []
    # Header row
    lines.append("| " + " | ".join(clean_headers) + " |")
    # Separator row
    lines.append("| " + " | ".join(clean_align) + " |")

    # Data rows
    for row in rows:
        clean_row = []
        for c_idx in range(num_cols):
            val = str(row[c_idx]) if c_idx < len(row) else ""
            clean_val = val.replace("|", "&#124;").strip()
            clean_row.append(clean_val)
        lines.append("| " + " | ".join(clean_row) + " |")

    return "\n\n" + "\n".join(lines) + "\n\n"


def repair_malformed_table_header(header_cell: str, empty_col_count: int) -> Optional[List[str]]:
    """
    Detects and repairs concatenated header cells like:
    'Change TypeAreaShareCluster CountPhysical Interpretation'
    returning individual column headers.
    """
    raw = header_cell.strip().strip("*").strip()
    # Known column vocabulary for Earth observation and GIS analysis
    known_tokens = [
        "Change Type", "Observed Signal", "Direction", "Category",
        "Area", "Footprint Area", "Share", "Landscape Share",
        "Cluster Count", "Clusters", "Region", "Candidate Cluster",
        "Physical Interpretation", "Centroid Location", "Location",
        "Coordinates", "Magnitude", "Index Operating Threshold",
        "Operating Threshold", "Operating Metric", "Visual / Morphological Signature",
        "Transition Status", "Statistical Evaluation", "Status", "Notes",
        "Plausible Category", "Confidence", "Confidence / Context",
        "Geospatial Context", "Details", "Primary Change", "Cluster ID", "Extent",
        "Scientific Evidence Factor", "Measured Value", "Quality Evaluation",
        "Brightening Metric", "Darkening Metric", "Agricultural Indicator",
        "Infrastructure Corridor Metric", "Spatial Geometry", "Rank"
    ]
    # Sort known tokens by length descending to match greedily
    known_tokens.sort(key=len, reverse=True)
    
    extracted = []
    remaining = raw
    while remaining:
        matched = False
        for tok in known_tokens:
            if remaining.lower().startswith(tok.lower()):
                extracted.append(tok)
                remaining = remaining[len(tok):].lstrip()
                matched = True
                break
        if not matched:
            # Try splitting camelCase / TitleCase word
            m = re.match(r"^([A-Z][a-z0-9]+(?:\s+[A-Za-z0-9]+)*)", remaining)
            if m:
                tok = m.group(1)
                extracted.append(tok)
                remaining = remaining[len(tok):].lstrip()
            else:
                # Fallback: take remaining
                extracted.append(remaining)
                break

    total_expected = 1 + empty_col_count
    if len(extracted) >= 2:
        return extracted
    return None


def sanitize_markdown(text: str) -> str:
    """
    Post-processes markdown text to ensure:
    1. Tables have blank lines before and after.
    2. Malformed tables with concatenated headers or mismatched column counts are repaired.
    3. Headings have a blank line before them.
    4. Code blocks and lists retain clean spacing.
    """
    if not text:
        return ""

    lines = text.split("\n")
    processed_lines = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Check if line looks like a table row (starts and ends with pipe, or has multiple pipes)
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
            # Gather consecutive table lines
            table_lines = [stripped]
            i += 1
            while i < n and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i].strip())
                i += 1

            # Process table block
            if len(table_lines) >= 2:
                first_cells = [c.strip() for c in table_lines[0].split("|")[1:-1]]
                
                # Check data row column counts (find max column count across data rows)
                data_col_counts = [len([c for c in r.split("|")[1:-1]]) for r in table_lines[2:]]
                max_data_cols = max(data_col_counts) if data_col_counts else 0

                # If first row only has 1 column or all remaining cells are empty, or data rows have more columns
                repaired_headers = None
                if len(first_cells) == 1 or (len(first_cells) >= 2 and all(c == "" for c in first_cells[1:])):
                    empty_count = max(len(first_cells) - 1, max_data_cols - 1)
                    repaired_headers = repair_malformed_table_header(first_cells[0], empty_count)
                elif max_data_cols > len(first_cells):
                    repaired_headers = repair_malformed_table_header(table_lines[0], max_data_cols - 1)

                if repaired_headers:
                    table_lines[0] = "| " + " | ".join(repaired_headers) + " |"
                    header_cols = len(repaired_headers)
                else:
                    header_cols = len(first_cells)

                # Check and repair separator row
                sep_cells = [c.strip() for c in table_lines[1].split("|")[1:-1]]
                is_valid_sep = len(sep_cells) == header_cols and all(
                    re.match(r"^:?-+:?$", c) for c in sep_cells if c
                )
                if not is_valid_sep:
                    # Provide sensible default alignment: right for area/share/counts, left otherwise
                    def_aligns = []
                    h_names = [c.strip().lower() for c in table_lines[0].split("|")[1:-1]]
                    for hn in h_names:
                        if any(term in hn for term in ("area", "share", "count", "magnitude", "ha", "%", "px")):
                            def_aligns.append("---:")
                        elif any(term in hn for term in ("rank", "status", "direction", "category")):
                            def_aligns.append(":---:")
                        else:
                            def_aligns.append(":---")
                    table_lines[1] = "| " + " | ".join(def_aligns) + " |"

                # Normalize all data rows to match header column count
                for r_idx in range(2, len(table_lines)):
                    r_cells = [c.strip() for c in table_lines[r_idx].split("|")[1:-1]]
                    if len(r_cells) < header_cols:
                        r_cells.extend([""] * (header_cols - len(r_cells)))
                    elif len(r_cells) > header_cols:
                        r_cells = r_cells[:header_cols]
                    table_lines[r_idx] = "| " + " | ".join(r_cells) + " |"

                # Ensure blank line before table if previous line isn't empty
                if processed_lines and processed_lines[-1].strip() != "":
                    processed_lines.append("")
                
                processed_lines.extend(table_lines)
                
                # Ensure blank line after table
                processed_lines.append("")
                continue
            else:
                processed_lines.extend(table_lines)
                continue

        # Ensure headings have a blank line before them
        if re.match(r"^#{1,6}\s+", stripped):
            if processed_lines and processed_lines[-1].strip() != "":
                processed_lines.append("")
            processed_lines.append(stripped)
            i += 1
            continue

        processed_lines.append(line)
        i += 1

    # Collapse any 3+ consecutive blank lines down to 2
    out = "\n".join(processed_lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()
