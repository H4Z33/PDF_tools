import unicodedata
import re
import json
import os
import time
import hashlib
from collections import defaultdict
from tqdm import tqdm
import fitz
import pdfplumber
import pandas as pd
import pymupdf4llm
from PIL import Image
import io
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import torch
from multiprocessing import Pool

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
    # Use deterministic algorithms if available, but avoid crashing if not
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        pass

# Shared session cache for Surya
FORMULA_CACHE = {}

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
RE_ABSTRACT = re.compile(r"^\s*[*_#]*\s*(abstract|resumen|summary)\b", re.IGNORECASE)
RE_KEYWORDS = re.compile(r"^\s*[*_#]*\s*(keywords|palabras\s*clave|key\s*words)\b", re.IGNORECASE)
RE_LIST_BULLET = re.compile(r"^\s*[-*•]\s+")
RE_LIST_ENUM = re.compile(r"^\s*\(?[a-zA-Z\d]+[\)\.]\s+")

RE_OMITTED_PIC = re.compile(r"^\*\*==> picture \[\d+ x \d+\] intentionally omitted <==\*\*")
RE_ISOLATED_META = re.compile(r"^(rev\.?\s*\d+|hoja:?\s*\d+|página\s*\d+|folio:?\s*\d+|pag\.?\s*\d+|hoja\s*#)$", re.IGNORECASE)
RE_L1_ANCHOR = re.compile(r"^[\s*_#\-]*1\.[\s]")
RE_DIGIT_START = re.compile(r"^[\s*_#\-]*\d+")
RE_L1_CANDIDATE = re.compile(r"^[\s*_#\-]*\d+\.[\s]+[A-Z]")
RE_FORMULA = re.compile(r'(?:[=+\-\u2212×x÷<>±∑∫√]|[\u03B1-\u03C9\u0391-\u03A9])[\s\S]*\(\d+(\.\d+)?\)\s*$', re.UNICODE)
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
# Reading Order & Layout
# ─────────────────────────────────────────
def detect_columns(blocks, k_max=3):
    if not blocks:
        return np.array([])
    xs = np.array([[ (b["bbox"][0] + b["bbox"][2]) / 2 ] for b in blocks])
    
    if len(xs) < 2:
        return np.zeros(len(xs))

    best_k = 1
    best_inertia = float("inf")
    best_model = None
    
    for k in range(1, min(k_max, len(xs)) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(xs)
        if km.inertia_ < best_inertia:
            best_k = k
            best_model = km
            best_inertia = km.inertia_
    
    return best_model.labels_

def sort_blocks(blocks):
    if not blocks:
        return []
    labels = detect_columns(blocks)
    
    enriched = []
    for b, col in zip(blocks, labels):
        # We use a tuple for sorting: (column, top, left)
        # Rounding top to 1 decimal place helps with slight alignment issues
        enriched.append((col, round(b["bbox"][1], 1), b["bbox"][0], b))
    
    # Sort by column → top → left
    enriched.sort(key=lambda x: (x[0], x[1], x[2]))
    
    return [e[-1] for e in enriched]


# ─────────────────────────────────────────
# Layout Graph & Clustering
# ─────────────────────────────────────────
def vertical_overlap(a_bbox, b_bbox):
    y0_a, y1_a = a_bbox[1], a_bbox[3]
    y0_b, y1_b = b_bbox[1], b_bbox[3]
    overlap = max(0, min(y1_a, y1_b) - max(y0_a, y0_b))
    h_a = y1_a - y0_a
    h_b = y1_b - y0_b
    if min(h_a, h_b) == 0: return 0
    return overlap / min(h_a, h_b)

def horizontal_gap(a_bbox, b_bbox):
    # Gap between right of a and left of b, or vice-versa
    if a_bbox[2] < b_bbox[0]: # a is left of b
        return b_bbox[0] - a_bbox[2]
    if b_bbox[2] < a_bbox[0]: # b is left of a
        return a_bbox[0] - b_bbox[2]
    return 0 # They overlap horizontally

def build_layout_graph(nodes):
    edges = []
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes[i+1:], i+1):
            # Connect if they have significant vertical overlap and small horizontal gap
            if vertical_overlap(a["bbox"], b["bbox"]) > 0.3:
                if horizontal_gap(a["bbox"], b["bbox"]) < 20:
                    # Also check font similarity if available
                    font_match = a.get("font") == b.get("font")
                    size_match = abs(a.get("size", 0) - b.get("size", 0)) < 1.0
                    if font_match or size_match:
                        edges.append((i, j))
    return edges

def cluster_nodes(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(edges)
    clusters = []
    for component in nx.connected_components(G):
        indices = sorted(list(component))
        cluster_nodes = [nodes[i] for i in indices]
        
        # Merge bounding box
        x0 = min(n["bbox"][0] for n in cluster_nodes)
        y0 = min(n["bbox"][1] for n in cluster_nodes)
        x1 = max(n["bbox"][2] for n in cluster_nodes)
        y1 = max(n["bbox"][3] for n in cluster_nodes)
        
        # Sort content by y then x
        cluster_nodes.sort(key=lambda n: (round(n["bbox"][1], 1), n["bbox"][0]))
        content = " ".join(n["text"] for n in cluster_nodes)
        
        clusters.append({
            "bbox": (x0, y0, x1, y1),
            "content": content,
            "nodes": cluster_nodes
        })
    return clusters


# ─────────────────────────────────────────
# Scoring & Classification
# ─────────────────────────────────────────
def score_block(content, norm, toc_map):
    scores = {
        "math": 0,
        "heading": 0,
        "list": 0,
        "paragraph": 1, # Base score
        "table": 0,
        "reference": 0
    }

    # 1. Math Scoring
    if RE_MATH_CHAR.search(content): scores["math"] += 3
    if "=" in content: scores["math"] += 2
    if "^" in content or "_" in content: scores["math"] += 2
    if RE_FORMULA.search(content): scores["math"] += 5

    # 2. Heading Scoring
    m_num = RE_HEADING_NUM.match(content)
    if m_num: scores["heading"] += 4
    if RE_HEADING_SINGLE.match(content): scores["heading"] += 3
    if RE_MARKDOWN_HEADING.match(content): scores["heading"] += 4
    if len(content) < 150 and not content.endswith("."): scores["heading"] += 1
    if norm in toc_map: scores["heading"] += 6

    # 3. List Scoring
    lines = content.split('\n')
    list_matches = sum(1 for ln in lines if RE_LIST_BULLET.match(ln) or RE_LIST_ENUM.match(ln))
    if list_matches > 1: scores["list"] += 5
    if list_matches == 1: scores["list"] += 2

    # 4. Table Scoring
    if "|" in content: scores["table"] += 5

    # 5. Caption / Special Types
    if RE_CAPTION.match(content):
        label = RE_CAPTION.match(content).group(1).lower()
        if label in ("table", "tabla", "cuadro"): return "table-caption"
        if label in ("figure", "figura"): return "figure-caption"
        return "illustration-caption"
    
    if RE_ABSTRACT.match(content): return "abstract"
    if RE_KEYWORDS.match(content): return "keywords"

    # Decide based on max score
    btype = max(scores, key=scores.get)
    
    # Refine heading level
    if btype == "heading":
        m_num = RE_HEADING_NUM.match(content)
        level = m_num.group(1).count(".") + 1 if m_num else 0
        if level == 0:
            m_md = RE_MARKDOWN_HEADING.match(content)
            level = len(m_md.group(1)) if m_md else 1
        return f"heading-l{level}"
        
    return btype


# ─────────────────────────────────────────
# Deep Learning Math Extractor
# ─────────────────────────────────────────
_surya_initialized = False

def _init_surya():
    """Lazy initialization of Surya predictor - called once."""
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
        # Attempt to pass device directly
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
                text = pred.text_lines[0].text if pred and pred.text_lines else None
                if text:
                    img_hash = hashlib.md5(batch_data[idx][0].tobytes()).hexdigest()
                    FORMULA_CACHE[img_hash] = text
                    results[idx] = text
        except Exception as e:
            print(f"Surya batch error: {e}")
            
    return results

def texify_on_demand(page, bbox, content, batch_container):
    """Pre-render formula image and add to batch for later batch processing."""
    rect = fitz.Rect(bbox)
    rect.x0 = max(0, rect.x0 - 10)
    rect.y0 = max(0, rect.y0 - 10)
    rect.x1 += 10
    rect.y1 += 10
    m = re.search(r'\((\d+(\.\d+)?)\)', content)
    num = m.group(1) if m else "?"
    try:
        pix = page.get_pixmap(clip=rect, dpi=192)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        # Store rect in img for later retrieval if possible, or just keep in batch
        batch_container.append((img, num, bbox))
    except Exception:
        pass
    return num


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

    # 3b. Abstract / Keywords
    if RE_ABSTRACT.match(content):
        return "abstract"
    m_kw = RE_KEYWORDS.match(content)
    if m_kw:
        return "keywords"

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
        # Support both raw and final schema
        b_type = b.get("block_type") or b.get("type") or ""
        if not b.get("dropped", False) and b_type.startswith("heading-l"):
            if RE_L1_ANCHOR.match(b["content"]):
                first_major_idx = i
                break
    
    if first_major_idx != -1:
        for i in range(first_major_idx):
            b = blocks[i]
            b_type = b.get("block_type") or b.get("type") or ""
            if not b.get("dropped", False) and b_type.startswith("heading-l"):
                if not RE_DIGIT_START.match(b["content"]):
                    if "block_type" in b: b["block_type"] = "heading-l0"
                    if "type" in b: b["type"] = "heading-l0"

    # 2. Upgrade Candidates based on Numbering Pattern consistency
    for b in blocks:
        b_type = b.get("block_type") or b.get("type") or ""
        if not b.get("dropped", False) and b_type in ("paragraph", "list-candidate"):
            content = b["content"].strip()
            # Hierarchical numbering check (N.N.N)
            m_num = RE_HEADING_NUM.match(content)
            if m_num:
                level = m_num.group(1).count(".") + 1
                new_type = f"heading-l{level}"
                if "block_type" in b: b["block_type"] = new_type
                if "type" in b: b["type"] = new_type
            # Level 1 check
            elif len(content) < 100 and RE_L1_CANDIDATE.match(content):
                new_type = "heading-l1"
                if "block_type" in b: b["block_type"] = new_type
                if "type" in b: b["type"] = new_type
                
    # 3. Contextual Reference Re-Classifier
    current_major_context = ""
    for b in blocks:
        if b.get("dropped", False):
            continue
            
        b_type = b.get("block_type") or b.get("type") or ""
        # Detect entry into the References section
        if b_type == "heading-l0" or b_type == "heading-l1":
            clean_str = re.sub(r'[^a-zA-Z]', '', b["content"].lower())
            if clean_str.startswith("referenc") or clean_str.startswith("bibliogr"):
                current_major_context = "references"
            elif b_type == "heading-l0":
                current_major_context = "body"
                
        # If inside References, lock down numbered arrays into reference types Instead of headings
        if current_major_context == "references":
            if b_type.startswith("heading-l") and not (re.sub(r'[^a-zA-Z]', '', b["content"].lower()).startswith("referenc")):
                if "block_type" in b: b["block_type"] = "reference"
                if "type" in b: b["type"] = "reference"
            elif re.match(r"^[\s*_#\-]*\[?\d+\]?[\.\s]", b["content"]):
                if "block_type" in b: b["block_type"] = "reference"
                if "type" in b: b["type"] = "reference"

    return blocks


# ─────────────────────────────────────────
# Pipeline Stages
# ─────────────────────────────────────────
def extract_primitives(page):
    """Stage 1: Extract raw spans and rectangles from PyMuPDF."""
    primitives = []
    try:
        dict_data = page.get_text("dict")
        for b in dict_data.get("blocks", []):
            if b["type"] != 0: continue # Skip images
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    primitives.append({
                        "bbox": span["bbox"],
                        "text": span["text"],
                        "font": span["font"],
                        "size": span["size"],
                        "color": span["color"],
                        "flags": span["flags"]
                    })
    except Exception as e:
        print(f"Error extracting primitives: {e}")
    return primitives

def process_math(page, blocks):
    """Stage 5: Detect and process math using Surya with batching."""
    formula_batch = []
    # Identify blocks that likely contain formulas
    for b in blocks:
        if RE_FORMULA.search(b["content"]):
            texify_on_demand(page, b["bbox"], b["content"], formula_batch)
    
    if formula_batch:
        latex_results = _surya_batch_process(formula_batch)
        results = []
        for i, latex in enumerate(latex_results):
            if latex:
                results.append({"bbox": formula_batch[i][2], "latex": latex})
        return results
    return []

# Helper for IOU
def calculate_iou(bbox1, bbox2):
    r1 = fitz.Rect(bbox1)
    r2 = fitz.Rect(bbox2)
    intersect = r1.intersect(r2)
    if intersect.is_empty: return 0
    area_intersect = intersect.width * intersect.height
    area_union = (r1.width * r1.height) + (r2.width * r2.height) - area_intersect
    return area_intersect / area_union if area_union > 0 else 0

def layout_entropy(blocks):
    """Calculate structural anomaly score based on y-coordinate variance."""
    if not blocks: return 0
    ys = [b["bbox"][1] for b in blocks]
    return np.std(ys)

def process_page(args):
    """Orchestrated pipeline for a single page."""
    pdf_path, page_idx, toc_map, route_llm_all = args
    page_num = page_idx + 1
    
    try:
        # Each process opens its own handle for safety
        doc = fitz.open(pdf_path)
        page = doc[page_idx]
        
        # Stage 1: Primitives
        primitives = extract_primitives(page)
        
        # Stage 2 & 3: Layout Graph & Clustering
        edges = build_layout_graph(primitives)
        blocks = cluster_nodes(primitives, edges)
        
        # Stage 4: Scoring & Classification
        for b in blocks:
            norm = normalize(b["content"])
            b["block_type"] = score_block(b["content"], norm, toc_map)
            b["dropped"] = False
            b["source"] = "rule"
        
        # Stage 5: Reading Order / Sorting
        blocks = sort_blocks(blocks)
        
        # Stage 6: LLM Routing Decision
        entropy = layout_entropy(blocks)
        route_to_llm = route_llm_all or (entropy > 50.0)
        
        if route_to_llm:
            mdfile = pymupdf4llm.to_markdown(pdf_path, text_mode=False, write_images=False, page_chunks=True, pages=[page_idx])
            if mdfile:
                text = mdfile[0].get("text", "")
                chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
                blocks = []
                for chunk in chunks:
                    norm = normalize(chunk)
                    blocks.append({
                        "bbox": [0,0,0,0],
                        "content": chunk,
                        "block_type": score_block(chunk, norm, toc_map),
                        "source": "llm",
                        "dropped": False
                    })
        else:
            # Stage 7: Math Refinement (Surya)
            math_results = process_math(page, blocks)
            for m in math_results:
                for b in blocks:
                    if calculate_iou(m["bbox"], b["bbox"]) > 0.3:
                        b["content"] = m["latex"]
                        b["block_type"] = "formula"
                        b["source"] = "ml"
                        
            # Stage 8: Table Extraction (pdfplumber)
            try:
                with pdfplumber.open(pdf_path) as doc_p:
                    page_p = doc_p.pages[page_idx]
                    tables = page_p.find_tables()
                    for tab in tables:
                        # Drop overlapping blocks
                        blocks = [b for b in blocks if calculate_iou(b["bbox"], tab.bbox) < 0.5]
                        
                        data = tab.extract()
                        if data and len(data) > 1:
                            df = pd.DataFrame(data[1:], columns=data[0])
                            df = df.replace('\n', ' ', regex=True)
                            md_table = df.to_markdown(index=False)
                            blocks.append({
                                "bbox": tab.bbox,
                                "content": md_table,
                                "block_type": "table",
                                "source": "rule",
                                "dropped": False
                            })
            except:
                pass
        
        doc.close()
        return finalize_output(blocks, page_num)
    except Exception as e:
        print(f"Error processing page {page_num}: {e}")
        return []

def finalize_output(blocks, page_num):
    """Stage 10: Standardize output schema."""
    final = []
    for i, b in enumerate(blocks):
        final.append({
            "page": page_num,
            "seq": i + 1,
            "type": b.get("block_type", "paragraph"),
            "bbox": b["bbox"],
            "content": b["content"],
            "meta": {
                "confidence": b.get("confidence", 1.0),
                "source": b.get("source", "rule")
            }
        })
    return final


# ─────────────────────────────────────────
# Filters & Post-processing
# ─────────────────────────────────────────
def frequency_filter(pages_blocks, threshold=0.05):
    """Identify headers/footers based on position and content consistency."""
    pos_content_counts = defaultdict(int)
    all_blocks = [b for p in pages_blocks for b in p]
    
    for b in all_blocks:
        pos_key = (round(b["bbox"][0], 0), round(b["bbox"][1], 0), normalize(b["content"]))
        pos_content_counts[pos_key] += 1
        
    num_pages = len(pages_blocks)
    cutoff = max(2, num_pages * threshold)
    
    for page in pages_blocks:
        for b in page:
            pos_key = (round(b["bbox"][0], 0), round(b["bbox"][1], 0), normalize(b["content"]))
            if pos_content_counts[pos_key] > cutoff:
                if not b["type"].startswith("heading"):
                    b["dropped"] = True
    
    return [[b for b in p if not b.get("dropped", False)] for p in pages_blocks]

def renumber(pages_blocks):
    seq = 1
    for page in pages_blocks:
        for b in page:
            b["seq"] = seq
            seq += 1
    return pages_blocks


# ─────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────
def run(pdf_path, out_path=None, max_pages=None, parallel=True):
    start_time = time.time()
    
    if out_path is None:
        out_path = os.path.splitext(pdf_path)[0] + "_ocr.json"
        
    toc_map = get_toc_map(pdf_path)
    try:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return
    
    num_to_process = min(max_pages, total_pages) if max_pages else total_pages
    args = [(pdf_path, i, toc_map, False) for i in range(num_to_process)]
    
    print(f"[*] Processing {num_to_process} pages (Parallel={parallel})...")
    
    if parallel and num_to_process > 1:
        with Pool() as pool:
            results = list(tqdm(pool.imap(process_page, args), total=num_to_process))
    else:
        results = [process_page(a) for a in tqdm(args)]
        
    # Apply Filters
    print("[*] Applying frequency filters and hierarchical refinement...")
    results = frequency_filter(results)
    
    # Flatten temporarily for global refinement
    flat_blocks = [b for p in results for b in p]
    # Set dropped flag based on existence in filtered lists
    # Actually frequency_filter already sets b["dropped"] = True
    
    refine_heading_hierarchy(flat_blocks)
    
    # Flatten and save
    results = renumber(results)
    final_output = [b for p in results for b in p]
    
    # Final cleanup (Markdown stripping etc)
    re_markdown = re.compile(r'[*_#]+')
    re_heading_num = re.compile(r'^[\s\-]*((?:Appendix\s+[A-Z]\.?|\d+(\.\d+)*)\.?)\s+', re.IGNORECASE)
    
    for b in final_output:
        if b["type"] not in ("table", "formula"):
            cleaned = b["content"]
            cleaned = re_markdown.sub('', cleaned)
            if b["type"].startswith("heading") or b["type"] == "reference":
                cleaned = re_heading_num.sub('', cleaned)
            b["content"] = " ".join(cleaned.split())
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    elapsed = time.time() - start_time
    print(f"\n[+] Processing complete in {elapsed:.2f}s")
    print(f"[+] Output saved to: {out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf-ocr.py <path_to_pdf>")
    else:
        run(sys.argv[1])
