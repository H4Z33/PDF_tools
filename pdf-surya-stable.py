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
from PIL import Image
import io
import hashlib
import torch
import numpy as np

# ─────────────────────────────────────────
# Torch & Device Setup
# ─────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

torch.manual_seed(0)
np.random.seed(0)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(0)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        pass

# Shared session cache for Surya
FORMULA_CACHE = {}

# ─────────────────────────────────────────
# Deep Learning Math Extractor
# ─────────────────────────────────────────
_surya_initialized = False

def _init_surya():
    global texify_predictor, TaskNames_cls, _surya_initialized
    if _surya_initialized:
        return
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.common.surya.schema import TaskNames
    TaskNames_cls = TaskNames
    print(f"\n\n[OCR] Surya Engine detected device: {DEVICE}")
    print(f"[OCR] Surya Convolutional Math Engine loading into {DEVICE}...")
    
    try:
        texify_predictor = RecognitionPredictor(FoundationPredictor(device=DEVICE))
    except Exception as e:
        print(f"[OCR] Surya initialization warning (falling back to auto): {e}")
        texify_predictor = RecognitionPredictor(FoundationPredictor())
        
    _surya_initialized = True

def _surya_batch_process(batch_data):
    """Process formula images in ONE Surya batch forward pass, with session-based caching."""
    _init_surya()
    if not batch_data:
        return []
        
    results = [None] * len(batch_data)
    to_process_indices = []
    to_process_images = []
    
    for i, (img, _) in enumerate(batch_data):
        img_hash = hashlib.md5(img.tobytes()).hexdigest()
        if img_hash in FORMULA_CACHE:
            results[i] = FORMULA_CACHE[img_hash]
        else:
            to_process_indices.append(i)
            to_process_images.append(img)
            
    if to_process_images:
        bboxes = [[[0, 0, img.width, img.height]] for img in to_process_images]
        try:
            preds = texify_predictor(
                to_process_images,
                [TaskNames_cls.block_without_boxes] * len(to_process_images),
                bboxes=bboxes
            )
            for idx, pred in zip(to_process_indices, preds):
                text = "\n".join([l.text for l in pred.text_lines]) if pred and pred.text_lines else None
                if text:
                    img_hash = hashlib.md5(batch_data[idx][0].tobytes()).hexdigest()
                    FORMULA_CACHE[img_hash] = text
                    results[idx] = text
        except Exception as e:
            print(f"Surya batch error: {e}")
            
    return results

def texify_on_demand(page, bbox, num, batch_container):
    """Pre-render formula image and add to batch for later batch processing."""
    rect = fitz.Rect(bbox)
    rect.x0 = max(0, rect.x0 - 10)
    rect.y0 = max(0, rect.y0 - 10)
    rect.x1 += 10
    rect.y1 += 10
    try:
        pix = page.get_pixmap(clip=rect, dpi=192)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        batch_container.append((img, num))
    except Exception:
        pass


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
RE_HEADING_SINGLE = re.compile(r"^[\s*_#\-]*(\d+)\.[\s]")
RE_MARKDOWN_HEADING = re.compile(r"^(#+)\s+")
RE_CAPTION = re.compile(r"^\s*[*_#]*\s*(table|tabla|cuadro|figure|figura|illustration|ilustracion|listing|algorithm)\b", re.IGNORECASE)
RE_LIST_BULLET = re.compile(r"^\s*[-*•]\s+")
RE_LIST_ENUM = re.compile(r"^\s*\(?[a-zA-Z\d]+[\)\.]\s+")

RE_OMITTED_PIC = re.compile(r"^\*\*==> picture \[\d+ x \d+\] intentionally omitted <==\*\*")
RE_ISOLATED_META = re.compile(r"^(rev\.?\s*\d+|hoja:?\s*\d+|página\s*\d+|folio:?\s*\d+|pag\.?\s*\d+|hoja\s*#)$", re.IGNORECASE)
RE_L1_ANCHOR = re.compile(r"^[\s*_#\-]*1\.[\s]")
RE_DIGIT_START = re.compile(r"^[\s*_#\-]*\d+")
RE_L1_CANDIDATE = re.compile(r"^[\s*_#\-]*\d+\.[\s]+[A-Z]")
RE_FORMULA = re.compile(r'(?:[=+\-\u2212×x÷<>±∑∫√ﬃ\ufb00-\ufb06]|[\u03B1-\u03C9\u0391-\u03A9])[\s\S]{0,600}[\(\[ð]\d+(\.\d+)?[\)\]]\s*$', re.UNICODE)
RE_MATH_CHAR = re.compile(r'[\u2200-\u22FF\u2190-\u21FF\u2A00-\u2AFF\u0370-\u03FF]')
RE_INLINE_SEED = re.compile(r'(?:[\u2200-\u22FF\u0370-\u03FF]|[\.\s][=<>±−×÷][\s\.]|[a-zA-Z]\s?[=<>±−×÷])')


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
            
    # 6. Formula Detection
    if RE_FORMULA.search(content):
        return "formula"

    # 7. Heading Resolution
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
                
    # 3. Contextual Reference Re-Classifier
    current_major_context = ""
    for b in blocks:
        if b["dropped"]:
            continue
            
        # Detect entry into the References section
        if b["block_type"] == "heading-l0" or b["block_type"] == "heading-l1":
            clean_str = re.sub(r'[^a-zA-Z]', '', b["content"].lower())
            if clean_str.startswith("referenc") or clean_str.startswith("bibliogr"):
                current_major_context = "references"
            elif b["block_type"] == "heading-l0":
                current_major_context = "body"
                
        # If inside References, lock down numbered arrays into reference types Instead of headings
        if current_major_context == "references":
            if b["block_type"].startswith("heading-l") and not (re.sub(r'[^a-zA-Z]', '', b["content"].lower()).startswith("referenc")):
                b["block_type"] = "reference"
            elif re.match(r"^[\s*_#\-]*\[?\d+\]?[\.\s]", b["content"]):
                b["block_type"] = "reference"

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
        
        # 1. Detect Tables on the current page using pdfplumber (Early Pass)
        page_p = doc_p.pages[i]
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

        # 2. Quick Scan for Invisible Table Captions and Native Formulas
        raw_text_blocks = [b for b in page.get_text("blocks") if b[6] == 0]
        has_caption = False
        
        # 1. Pre-pass: Precision Visual Capture (Radar + Cropping)
        formula_batch = []
        page_formulas = []
        try:
            page_dict = page.get_text("dict")
            for j, b in enumerate(page_dict["blocks"]):
                if b["type"] != 0: continue
                
                # Reconstruct text for formula matching
                full_text = ""
                for l in b["lines"]:
                    line_text = "".join([s["text"] for s in l["spans"]])
                    full_text += line_text + "\n"
                
                match = RE_FORMULA.search(full_text)
                if match:
                    if intersects_any_table(b["bbox"]): continue
                    
                    # Compute focused bbox of ONLY the lines within the block that actually contain math
                    curr_idx = 0
                    focused_bbox = None
                    for l in b["lines"]:
                        line_text = "".join([s["text"] for s in l["spans"]]) + "\n"
                        line_len = len(line_text)
                        
                        # If this line overlaps with the match suffix
                        if curr_idx + line_len > match.start():
                            if focused_bbox is None:
                                focused_bbox = list(l["bbox"])
                            else:
                                focused_bbox[0] = min(focused_bbox[0], l["bbox"][0])
                                focused_bbox[1] = min(focused_bbox[1], l["bbox"][1])
                                focused_bbox[2] = max(focused_bbox[2], l["bbox"][2])
                                focused_bbox[3] = max(focused_bbox[3], l["bbox"][3])
                        curr_idx += line_len
                    
                    if focused_bbox:
                        # Radar logic: If block above has operators, merge it too
                        if j > 0:
                            prev_b = page_dict["blocks"][j-1]
                            if prev_b["type"] == 0:
                                prev_text = "".join(["".join([s["text"] for s in l["spans"]]) for l in prev_b["lines"]])
                                if any(c in prev_text for c in "=+\u2212×÷±") and not intersects_any_table(prev_b["bbox"]):
                                    focused_bbox[0] = min(focused_bbox[0], prev_b["bbox"][0])
                                    focused_bbox[1] = min(focused_bbox[1], prev_b["bbox"][1])
                                    focused_bbox[2] = max(focused_bbox[2], prev_b["bbox"][2])
                                    focused_bbox[3] = max(focused_bbox[3], prev_b["bbox"][3])

                        m = re.search(r'\((\d+(\.\d+)?)\)', full_text)
                        num = m.group(1) if m else "?"
                        texify_on_demand(page, focused_bbox, num, formula_batch)
        except: pass

        if formula_batch:
            latex_results = _surya_batch_process(formula_batch)
            for i, latex in enumerate(latex_results):
                if latex:
                    page_formulas.append({"num": formula_batch[i][1], "text": latex})


        # 2. Programmatic Inline Math Detection
        # We group adjacent spans on the same line that are mathy or scripts to find replacements
        inline_replacements = []
        try:
            dict_data = page.get_text("dict")
            for b in dict_data.get("blocks", []):
                if b["type"] != 0: continue
                all_sizes = [round(s["size"], 1) for l in b["lines"] for s in l["spans"]]
                if not all_sizes: continue
                dom_size = max(set(all_sizes), key=all_sizes.count)
                
                for line in b.get("lines", []):
                    origins = [round(s["origin"][1], 1) for s in line.get("spans", [])]
                    base_y = max(set(origins), key=origins.count) if origins else 0
                    current_raw, current_latex, is_in_math = [], [], False
                    
                    for s in line.get("spans", []):
                        text, streq = s["text"], s["text"].strip()
                        if not streq:
                            if is_in_math: current_raw.append(text); current_latex.append(" ")
                            continue
                        
                        f_lower = s["font"].lower()
                        is_math_font = any(f in f_lower for f in ("math", "symbol", "cmsy", "cmmi", "pazo"))
                        is_math_char = RE_MATH_CHAR.search(streq) is not None
                        is_script = (s["size"] < dom_size * 0.85)
                        
                        if is_math_font or is_math_char or is_script:
                            is_in_math = True
                            current_raw.append(text)
                            val = streq
                            if is_script:
                                dy = s["origin"][1] - base_y
                                if dy < -1: val = f"^{{{streq}}}" if "^" not in streq else streq
                                elif dy > 1: val = f"_{{{streq}}}" if "_" not in streq else streq
                            current_latex.append(val)
                        else:
                            if is_in_math:
                                raw_seq, lat_seq = "".join(current_raw).strip(), "".join(current_latex).strip()
                                if len(raw_seq) > 0 and (RE_MATH_CHAR.search(raw_seq) or "^" in lat_seq or "_" in lat_seq):
                                    inline_replacements.append((raw_seq, f"${lat_seq}$"))
                                current_raw, current_latex, is_in_math = [], [], False
                    if is_in_math:
                        raw_seq, lat_seq = "".join(current_raw).strip(), "".join(current_latex).strip()
                        if len(raw_seq) > 0 and (RE_MATH_CHAR.search(raw_seq) or "^" in lat_seq or "_" in lat_seq):
                            inline_replacements.append((raw_seq, f"${lat_seq}$"))
        except: pass
        
        final_repls, seen_raw = [], set()
        for raw, lat in sorted(inline_replacements, key=lambda x: len(x[0]), reverse=True):
            if raw in seen_raw or (len(raw) == 1 and not RE_MATH_CHAR.search(raw)): continue
            final_repls.append((raw, lat)); seen_raw.add(raw)
        inline_replacements = final_repls
                
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
                form_idx = 0
                for b in raw_chunks:
                    has_math = False
                    
                    # 1. Notation Restoration Fallback (Surgical Surya Injection)
                    if RE_OMITTED_PIC.match(b) and form_idx < len(page_formulas):
                        b = page_formulas[form_idx]["text"]
                        has_math = True
                        form_idx += 1
                    else:
                        match = RE_FORMULA.search(b)
                        if match:
                            m_num = re.search(r'\((\d+(\.\d+)?)\)', b)
                            num = m_num.group(1) if m_num else "?"
                            matched = False
                            for f in page_formulas:
                                if f["num"] == num:
                                    # Hybrid Replacement: Only replace the matching math anchor segment
                                    prefix = b[:match.start()]
                                    b = prefix.strip() + "\n" + f["text"]
                                    has_math = True
                                    matched = True
                                    break
                            if not matched and form_idx < len(page_formulas):
                               prefix = b[:match.start()]
                               b = prefix.strip() + "\n" + page_formulas[form_idx]["text"]
                               has_math = True
                               form_idx += 1
                    
                    # 2. Fuzzy Inline Math Repair (Programmatic)
                    if not has_math and RE_INLINE_SEED.search(b):
                        for raw, lat in inline_replacements:
                            pattern = re.escape(raw).replace(r'\ ', r'\s*')
                            if re.search(pattern, b):
                                b = re.sub(pattern, lat, b)
                                has_math = True
                        while "$$" in b: b = b.replace("$$", "$")
                    
                    norm = normalize(b)
                    is_paging = False
                    if not norm.strip():
                        is_paging = True
                    elif b.isdigit() and len(b) < 4:
                        is_paging = True
                    elif RE_ISOLATED_META.match(b):
                        is_paging = True
                        
                    # Anti-outlier check
                    if has_math and ("Table" in b or "|" in b or b.count('\n') > 10):
                        has_math = False

                    btype = "formula" if has_math else identify_block_type(b, norm, toc_map)
                    
                    blocks.append({
                        "page": page_num,
                        "content": b,
                        "raw": b,
                        "block_type": btype,
                        "dropped": is_paging,
                        "drop_reason": "paging" if is_paging else None,
                    })
            continue # Processed via LLM, skip the fast engines for this page!

        # 3. STANDARD FAST ENGINE
        # Tables were already detected in the Early Pass

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
            form_idx = 0 # Need to track this here too
            
            for b in raw_blocks:
                has_math = False
                if RE_OMITTED_PIC.match(b):
                    # In fast branch this is rare but possible if pictures are enabled
                    if form_idx < len(page_formulas):
                        b = page_formulas[form_idx]["text"]
                        has_math = True
                        form_idx += 1
                    else: continue
                    
                # Notation Restoration Radar for Fast Branch (Surgical)
                match = RE_FORMULA.search(b)
                if match:
                    m_num = re.search(r'\((\d+(\.\d+)?)\)', b)
                    num = m_num.group(1) if m_num else "?"
                    matched = False
                    for f in page_formulas:
                        if f["num"] == num:
                            prefix = b[:match.start()]
                            b = prefix.strip() + "\n" + f["text"]
                            has_math = True
                            matched = True
                            break
                    if not matched and form_idx < len(page_formulas):
                        prefix = b[:match.start()]
                        b = prefix.strip() + "\n" + page_formulas[form_idx]["text"]
                        has_math = True
                        form_idx += 1
                elif not has_math and RE_INLINE_SEED.search(b):
                     for raw, lat in inline_replacements:
                         pattern = re.escape(raw).replace(r'\ ', r'\s*')
                         if re.search(pattern, b):
                             b = re.sub(pattern, lat, b)
                             has_math = True
                     while "$$" in b: b = b.replace("$$", "$")

                norm = normalize(b)
                # Comprehensive Paging/Metadata check
                is_paging = False
                if not norm.strip():
                    is_paging = True
                elif b.isdigit() and len(b) < 4:
                    is_paging = True
                elif RE_ISOLATED_META.match(b):
                    is_paging = True

                # Anti-outlier check
                if has_math and ("Table" in b or "|" in b or b.count('\n') > 10):
                    has_math = False

                btype = "formula" if has_math else identify_block_type(b, norm, toc_map)

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
        out_path = os.path.splitext(pdf_path)[0] + "_surya.json"

    # Final cleanup of metadata fields
    final_output = []
    
    # Precompile regex for fast final cleanup
    re_markdown = re.compile(r'[*_#]+')
    re_heading_num = re.compile(r'^[\s\-]*((?:Appendix\s+[A-Z]\.?|\d+(\.\d+)*)\.?)\s+', re.IGNORECASE)
    
    for b in blocks:
        if not b["dropped"] and b["block_type"] not in ("table", "formula"):
            cleaned = b["content"]
            # 1. Strip structural markdown (bold, italic, headers)
            cleaned = re_markdown.sub('', cleaned)
            # 2. Strip redundant numbering from headings
            if b["block_type"].startswith("heading") or b["block_type"] == "reference":
                cleaned = re_heading_num.sub('', cleaned)

            b["content"] = " ".join(cleaned.split())
            
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
    pdf = sys.argv[1]
    pages = None
    if "--max-pages" in sys.argv:
        idx = sys.argv.index("--max-pages")
        pages = int(sys.argv[idx+1])
    run(pdf, max_pages=pages)
