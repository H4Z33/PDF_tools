import unicodedata
import re
import json
import os
import time
from collections import defaultdict
from tqdm import tqdm
import fitz
import pdfplumber
import pandas as pd
import pymupdf4llm

# ─────────────────────────────────────────
# Constants & Compiled Regexes
# ─────────────────────────────────────────
SUP_CHARS = set("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ")
SUB_CHARS = set("₀₁₂₃₄₅₆₇₈₉")

# Pre-compile regexes for significant speedup
RE_PAGING_METADATA = re.compile(r'\b(page|p\.?|pg|pagina|pag|pag\.|pág|pág\.|hoja|rev\.?|rev|folio)\b\s*:?\s*\d+(\s*(of|de|/)\s*\d+)?\b', re.IGNORECASE)
RE_PAGING_FRACTION = re.compile(r'\b\d+\s*(of|de|/|de)\s*\d+\b', re.IGNORECASE)
RE_NUMBERS = re.compile(r'\b\d+\b')
RE_NON_WORD = re.compile(r'[^\w\s]')

RE_HEADING_NUM = re.compile(r"^[\s*_#\-]*(\d+(\.\d+)+)\b")
RE_HEADING_SINGLE = re.compile(r"^[\s*_#\-]*(\d+)[\.\s]")
RE_MARKDOWN_HEADING = re.compile(r"^(#+)\s+")
RE_CAPTION = re.compile(r"^\s*[*_#]*\s*(table|tabla|cuadro|figure|figura|illustration|ilustracion)\b", re.IGNORECASE)
RE_LIST_BULLET = re.compile(r"^\s*[-*•]\s+")
RE_LIST_ENUM = re.compile(r"^\s*\(?[a-zA-Z\d]+[\)\.]\s+")

RE_OMITTED_PIC = re.compile(r"^\*\*==> picture \[\d+ x \d+\] intentionally omitted <==\*\*")
RE_ISOLATED_META = re.compile(r"^(rev\.?\s*\d+|hoja:?\s*\d+|página\s*\d+|folio:?\s*\d+|pag\.?\s*\d+|hoja\s*#)$", re.IGNORECASE)
RE_L1_ANCHOR = re.compile(r"^[\s*_#\-]*1[\.\s]")
RE_DIGIT_START = re.compile(r"^[\s*_#\-]*\d+")
RE_L1_CANDIDATE = re.compile(r"^[\s*_#\-]*\d+[\.\s]+[A-Z]")


# ─────────────────────────────────────────
# Normalization (comparison only)
# ─────────────────────────────────────────
def normalize(text: str) -> str:
    # 1. Base normalization (NFC + Lowercase)
    text = unicodedata.normalize("NFKC", text).lower()

    # 2. De-accent (Strip marks like 'á' -> 'a')
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    # 3. Paging and Metadata patterns
    text = RE_PAGING_METADATA.sub(' ', text)
    text = RE_PAGING_FRACTION.sub(' ', text)
    
    # Generic numbers
    text = RE_NUMBERS.sub(' ', text)

    # 4. Filter symbols and collapse whitespace
    text = RE_NON_WORD.sub(' ', text)
    return text.strip()


# ─────────────────────────────────────────
# Detection Helpers
# ─────────────────────────────────────────
def get_toc_map(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()
        doc.close()
        return {normalize(title): level for level, title, page in toc}
    except:
        return {}


# ─────────────────────────────────────────
# Hierarchical Refinement
# ─────────────────────────────────────────
def identify_block_type(content, norm, toc_map):
    # 1. Hierarchical Numbering (e.g. 2.2.2)
    m_num = RE_HEADING_NUM.match(content)
    d_num = m_num.group(1).count(".") + 1 if m_num else 0

    # Numbering like "1. Introduction" or "1 Introduction"
    m_single_num = RE_HEADING_SINGLE.match(content)
    d_single = 1 if m_single_num else 0

    # 2. Markdown Depth (# Heading) -> rarely triggered in pure fitz
    m_md = RE_MARKDOWN_HEADING.match(content)
    d_md = len(m_md.group(1)) if m_md else 0

    # 3. Specific Captions
    m_cap = RE_CAPTION.match(content)
    if m_cap:
        label = m_cap.group(1).lower()
        if label in ("table", "tabla", "cuadro"):
            return "table-caption"
        elif label in ("figure", "figura"):
            return "figure-caption"
        return "illustration-caption"

    # 4. Table / ToC Matching
    if "|" in content:
        return "table"
    if norm in toc_map:
        return f"heading-l{toc_map[norm]}"

    # 5. List detection (PRIORITY OVER CANDIDATE)
    lines = content.split('\n')
    list_matches = sum(1 for ln in lines if RE_LIST_BULLET.match(ln) or RE_LIST_ENUM.match(ln))
    
    if list_matches > 1:
        return "list"
    if list_matches == 1:
        # Check if it's a numbered heading first
        if d_num == 0 and d_single == 0:
            return "list-candidate"

    # 6. Heading Resolution
    if d_num > 0:
        return f"heading-l{d_num}"
    if d_md > 0:
        return f"heading-l{d_md}"
    if d_single > 0 and len(content) < 150:
        return "heading-l1"

    # 7. Candidate (Un-numbered context)
    is_cand = (len(content) < 150 and not content.endswith(".")) or content.endswith(":")
    if is_cand:
        return "list-candidate" if ":" in content else "paragraph"

    return "paragraph"


def refine_heading_hierarchy(blocks):
    """
    Second pass: Global hierarchical adjustment.
    """
    # 1. Title Promotion (Pre-Introduction l0)
    first_major_idx = -1
    for i, b in enumerate(blocks):
        if not b["dropped"] and b["block_type"].startswith("heading-l"):
            if RE_L1_ANCHOR.match(b["content"]):
                first_major_idx = i
                break
    
    if first_major_idx != -1:
        for i in range(first_major_idx):
            b = blocks[i]
            if not b["dropped"] and b["block_type"].startswith("heading-l"):
                if not RE_DIGIT_START.match(b["content"]):
                    b["block_type"] = "heading-l0"

    # 2. Upgrade Candidates based on Numbering Pattern consistency
    for b in blocks:
        if not b["dropped"] and b["block_type"] in ("paragraph", "list-candidate"):
            content = b["content"].strip()
            # Hierarchical numbering check (N.N.N)
            m_num = RE_HEADING_NUM.match(content)
            if m_num:
                level = m_num.group(1).count(".") + 1
                b["block_type"] = f"heading-l{level}"
            # Level 1 check
            elif len(content) < 100 and RE_L1_CANDIDATE.match(content):
                b["block_type"] = "heading-l1"
                
    return blocks


# ─────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────
def extract_blocks(pdf_path, max_pages=None):
    toc_map = get_toc_map(pdf_path)
    blocks = []
    
    table_count = 0
    t_table_ext = 0
    t_llm_ext = 0
    pages_llm = 0
    llm_linger = 0
    
    doc = fitz.open(pdf_path)
    doc_p = pdfplumber.open(pdf_path)
    
    if max_pages is not None:
        pages_to_process = range(min(max_pages, doc.page_count))
    else:
        pages_to_process = range(doc.page_count)

    for i in pages_to_process:
        page = doc[i]
        page_num = i + 1
        
        # 1. Quick Scan for Invisible Table Captions
        raw_text_blocks = [b for b in page.get_text("blocks") if b[6] == 0]
        has_caption = False
        for b in raw_text_blocks:
            if RE_CAPTION.match(b[4]):
                has_caption = True
                break
                
        route_to_llm = False
        if has_caption:
            route_to_llm = True
            llm_linger = 1  # Route the next page too to catch spillover
        elif llm_linger > 0:
            route_to_llm = True
            llm_linger -= 1
            
        if route_to_llm:
            pages_llm += 1
            t_llm0 = time.time()
            mdfile = pymupdf4llm.to_markdown(pdf_path, text_mode=False, write_images=False, page_chunks=True, pages=[i])
            t_llm_ext += time.time() - t_llm0
            
            if mdfile:
                text = mdfile[0].get("text", "")
                # Extrapolate tables found dynamically from markdown markers
                table_count += text.count("|---")
                
                raw_chunks = [b.strip() for b in text.split("\n\n") if b.strip()]
                for b in raw_chunks:
                    if RE_OMITTED_PIC.match(b):
                        continue
                    
                    norm = normalize(b)
                    is_paging = False
                    if not norm.strip():
                        is_paging = True
                    elif b.isdigit() and len(b) < 4:
                        is_paging = True
                    elif RE_ISOLATED_META.match(b):
                        is_paging = True
                        
                    btype = identify_block_type(b, norm, toc_map)
                    
                    blocks.append({
                        "page": page_num,
                        "content": b,
                        "raw": b,
                        "block_type": btype,
                        "dropped": is_paging,
                        "drop_reason": "paging" if is_paging else None,
                    })
            continue # Processed via LLM, skip the fast engines for this page!

        # 2. STANDARD FAST ENGINE
        page_p = doc_p.pages[i]
        
        # 1. Detect Tables on the current page using pdfplumber
        t_tab0 = time.time()
        tables = page_p.find_tables()
        t_table_ext += time.time() - t_tab0
        
        table_rects = [tab.bbox for tab in tables]
        table_count += len(table_rects)
        
        def intersects_any_table(b_rect):
            r1 = fitz.Rect(b_rect)
            for t_bbox in table_rects:
                if r1.intersects(fitz.Rect(t_bbox)):
                    return True
            return False

        # 2. Extract standard text blocks
        # get_text("blocks") returns tuples: (x0, y0, x1, y1, "text", block_no, block_type)
        # We only want pure text that doesn't overlap with a table.
        page_blocks = [b for b in page.get_text("blocks") if b[6] == 0]
        
        for b_data in page_blocks:
            if intersects_any_table(b_data[:4]):
                continue  # Let the table extractor handle this text
                
            text = b_data[4]
            
            # Split by double newline as requested
            raw_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
            
            for b in raw_blocks:
                if RE_OMITTED_PIC.match(b):
                    continue

                norm = normalize(b)
                # Comprehensive Paging/Metadata check
                is_paging = False
                if not norm.strip():
                    is_paging = True
                elif b.isdigit() and len(b) < 4:
                    is_paging = True
                elif RE_ISOLATED_META.match(b):
                    is_paging = True

                btype = identify_block_type(b, norm, toc_map)

                blocks.append({
                    "page": page_num,
                    "content": b,
                    "raw": b,
                    "block_type": btype,
                    "dropped": is_paging,
                    "drop_reason": "paging" if is_paging else None,
                })
        
        # 3. Append Tables to the end of the page blocks
        for tab in tables:
            data = tab.extract()
            if data and len(data) > 1:
                header = data[0]
                rows = data[1:]
                
                # Make header unique for pandas DataFrame
                seen = {}
                new_header = []
                for h in header:
                    clean_h = str(h).replace('\n', ' ').strip() if h else ""
                    if clean_h in seen:
                        seen[clean_h] += 1
                        new_header.append(f"{clean_h} ({seen[clean_h]})")
                    else:
                        seen[clean_h] = 0
                        new_header.append(clean_h)
                
                try:
                    df = pd.DataFrame(rows, columns=new_header)
                    df = df.replace('\n', ' ', regex=True)
                    md_table = df.to_markdown(index=False)
                    
                    blocks.append({
                        "page": page_num,
                        "content": md_table,
                        "raw": md_table,
                        "block_type": "table",
                        "dropped": False,
                        "drop_reason": None,
                    })
                except Exception:
                    pass
                
    doc.close()
    doc_p.close()
    return blocks, toc_map, table_count, t_table_ext, t_llm_ext, pages_llm


# ─────────────────────────────────────────
# Filters
# ─────────────────────────────────────────
def frequency_filter(blocks, threshold=0.05):
    text_pages = defaultdict(set)
    norm_cache = {}

    for b in tqdm(blocks, desc="Frequency (Analyze)", leave=False):
        raw = b["raw"]
        if raw not in norm_cache:
            norm_cache[raw] = normalize(raw)
        norm = norm_cache[raw]
        text_pages[norm].add(b["page"])

    total_pages = max(b["page"] for b in blocks)
    # Cutoff floor of 2: prevents dropping things that only appear 1 or 2 times
    cutoff = max(2, total_pages * threshold)

    first_appearance_kept = set()

    for b in tqdm(blocks, desc="Frequency (Filter)", leave=False):
        if b["dropped"]:
            continue
        norm = norm_cache[b["raw"]]
        
        # Strict Protection Rule: Only protect hierarchical headings (## or N.N)
        is_protected = False
        if b["block_type"].startswith("heading-l"):
            if "##" in b["content"] or RE_HEADING_NUM.search(b["content"]):
                is_protected = True
        
        if len(text_pages[norm]) > cutoff and not is_protected:
            if norm not in first_appearance_kept:
                # Keep the very first appearance
                first_appearance_kept.add(norm)
            else:
                b["dropped"] = True
                b["drop_reason"] = "frequency"

    return blocks


# ─────────────────────────────────────────
# Renumber
# ─────────────────────────────────────────
def renumber(blocks):
    seq = 1
    for b in blocks:
        if not b["dropped"]:
            b["seq"] = seq
            seq += 1
        else:
            b["seq"] = 0
    return blocks


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def run(pdf_path, out_path="output.json", max_pages=None):
    stats = []
    start_total = time.time()

    # 1. Extraction (and initial paging filter)
    t0 = time.time()
    blocks, toc_map, n_tabs, t_tabs, t_llm, p_llm = extract_blocks(pdf_path, max_pages=max_pages)
    t_ext = time.time() - t0 - t_tabs - t_llm
    stats.append(("Text Extraction", t_ext, sum(1 for b in blocks if b["dropped"]), 0))
    stats.append(("Table Engine (pdfplumber)", t_tabs, 0, n_tabs))
    stats.append((f"LLM Engine ({p_llm} Pages)", t_llm, 0, 0))

    # 2. Frequency Filter
    t0 = time.time()
    blocks = frequency_filter(blocks, threshold=0.05)
    t_freq = time.time() - t0
    stats.append(("Frequency Filter", t_freq, sum(1 for b in blocks if b["dropped"] and b["drop_reason"] == "frequency"), 0))

    # 3. Structural Refinement (Second Pass)
    t0 = time.time()
    blocks = refine_heading_hierarchy(blocks)
    t_ref = time.time() - t0
    stats.append(("Structural Refinement", t_ref, 0, 0))

    # 4. Renumbering & Finalizing
    t0 = time.time()
    blocks = renumber(blocks)
    
    if out_path == "output.json":
        out_path = os.path.splitext(pdf_path)[0] + "_ext.json"

    # Final cleanup of metadata fields
    final_output = []
    for b in blocks:
        clean_b = {k: v for k, v in b.items() if k not in ("raw", "drop_reason")}
        final_output.append(clean_b)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    t_save = time.time() - t0
    stats.append(("Renumbering & Save", t_save, 0, 0))

    total_time = time.time() - start_total
    total_dropped = sum(s[2] for s in stats)
    total_tables = sum(s[3] for s in stats)

    # Print Summary Table
    print("\n" + "="*80)
    print(f"| {'PIPELINE SUMMARY (PURE FITZ)':^76} |")
    print("="*80)
    print(f"| {'Phase':<30} | {'Time (s)':<12} | {'Dropped':<10} | {'Tables':<10} |")
    print("-" * 80)
    for phase, duration, dropped, tabs in stats:
        t_str = str(tabs) if tabs > 0 else "-"
        print(f"| {phase:<30} | {duration:>12.3f} | {dropped:>10} | {t_str:>10} |")
    print("-" * 80)
    print(f"| {'TOTAL':<30} | {total_time:>12.3f} | {total_dropped:>10} | {total_tables:>10} |")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    run(sys.argv[1])
