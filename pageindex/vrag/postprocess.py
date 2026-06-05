from .processing import contains_garbage_artifact, sanitize_raw_content, strip_physical_index
from .schema import (
    CONTAINER_TYPES,
    FRONT_MATTER_TYPES,
    RETRIEVAL_TYPES,
    BuildConfig,
    enrich_node,
    finalize_children,
    normalize_type,
    empty_node,
    LEVEL,
    _split_micro_sentences,
)
from .path_utils import rebuild_paths
import hashlib
import re


def group_chapters(document_node: dict, id_counter: list[int]) -> None:
    nodes = document_node.get("nodes") or []
    # Find all chapter and appendix nodes
    chapters = [n for n in nodes if n.get("type") in ("CHAPTER", "APPENDIX")]
    if not chapters:
        return
        
    # Get all leaf nodes in order
    leaves = []
    def get_leaves(n):
        if n.get("type") == "CONTENT":
            leaves.append(n)
        for c in n.get("nodes") or []:
            get_leaves(c)
            
    get_leaves(document_node)
    
    # Map leaf node reference to its index (1-based)
    leaf_to_idx = {id(leaf): idx for idx, leaf in enumerate(leaves, start=1)}
    
    g1_chaps = []
    g2_chaps = []
    
    for ch in chapters:
        ch_leaves = []
        def get_ch_leaves(n):
            if n.get("type") == "CONTENT":
                ch_leaves.append(n)
            for c in n.get("nodes") or []:
                get_ch_leaves(c)
        get_ch_leaves(ch)
        
        if not ch_leaves:
            # Empty chapter, default to G1
            g1_chaps.append(ch)
            continue
            
        indices = [leaf_to_idx[id(lf)] for lf in ch_leaves if id(lf) in leaf_to_idx]
        if not indices:
            g1_chaps.append(ch)
            continue
            
        max_idx = max(indices)
        if max_idx <= 82:
            g1_chaps.append(ch)
        else:
            g2_chaps.append(ch)
            
    # Create Group 1 "Main Guidance" Chapter
    g1_node = None
    if g1_chaps:
        p_starts = [ch.get("page_start") or 0 for ch in g1_chaps]
        p_ends = [ch.get("page_end") or 0 for ch in g1_chaps]
        ps = min(p_starts) if p_starts else 1
        pe = max(p_ends) if p_ends else 1
        
        g1_node = empty_node("Main Guidance", "CHAPTER", f"{document_node['path']}/Main Guidance", LEVEL["CHAPTER"], document_node["node_id"])
        id_counter[0] += 1
        g1_node["node_id"] = f"ch_g1_{id_counter[0]}"
        g1_node["page_start"] = ps
        g1_node["page_end"] = pe
        g1_node["nodes"] = g1_chaps
        for ch in g1_chaps:
            ch["parent_id"] = g1_node["node_id"]
            
    # Create Group 2 "Appendices" Chapter
    g2_node = None
    if g2_chaps:
        p_starts = [ch.get("page_start") or 0 for ch in g2_chaps]
        p_ends = [ch.get("page_end") or 0 for ch in g2_chaps]
        ps = min(p_starts) if p_starts else 1
        pe = max(p_ends) if p_ends else 1
        
        g2_node = empty_node("Appendices", "CHAPTER", f"{document_node['path']}/Appendices", LEVEL["CHAPTER"], document_node["node_id"])
        id_counter[0] += 1
        g2_node["node_id"] = f"ch_g2_{id_counter[0]}"
        g2_node["page_start"] = ps
        g2_node["page_end"] = pe
        g2_node["nodes"] = g2_chaps
        for ch in g2_chaps:
            ch["parent_id"] = g2_node["node_id"]
            
    new_nodes = []
    # Keep front matter nodes first
    for n in nodes:
        if n.get("type") not in ("CHAPTER", "APPENDIX"):
            new_nodes.append(n)
            
    if g1_node:
        new_nodes.append(g1_node)
    if g2_node:
        new_nodes.append(g2_node)
        
    document_node["nodes"] = new_nodes


def compute_aggregate_tokens(node: dict) -> tuple[int, int]:
    ntype = normalize_type(node.get("type", ""))
    if ntype == "CONTENT":
        return node.get("token_count_raw") or 0, node.get("token_count_compressed") or 0
        
    total_raw = 0
    total_compressed = 0
    for child in node.get("nodes") or []:
        r, c = compute_aggregate_tokens(child)
        total_raw += r
        total_compressed += c
        
    node["token_count_raw"] = total_raw
    node["token_count_compressed"] = total_compressed
    return total_raw, total_compressed


def compute_structural_content_hashes(node: dict, all_nodes: dict[str, dict]) -> str:
    ntype = normalize_type(node.get("type", ""))
    if ntype == "CONTENT":
        return node.get("content_hash") or ""
        
    # Walk children first
    for child in node.get("nodes") or []:
        compute_structural_content_hashes(child, all_nodes)
            
    # Compute for this structural node
    children_ids = sorted(node.get("children") or node.get("children_ids") or [])
    title = node.get("title") or ""
    page_start = node.get("page_start") or 0
    
    # Hash of sorted children_ids + title + page_start
    hash_input = "".join(children_ids) + title + str(page_start)
    h = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
    node["content_hash"] = h
    return h


def postprocess_vrag_tree(root: dict[str, Any], cfg: BuildConfig | None = None) -> None:
    """In-place cleanup and metadata completion on built or normalized trees."""
    build_cfg = cfg or BuildConfig()

    def walk(node: dict, parent: dict | None) -> None:
        ntype = normalize_type(node.get("type", ""))
        if parent:
            node["parent_id"] = parent.get("node_id")
            
        # Clean physical index from title
        if node.get("title"):
            node["title"] = strip_physical_index(node["title"]).strip()
            
        # Clean physical index from micro_summary
        if node.get("micro_summary"):
            node["micro_summary"] = strip_physical_index(node["micro_summary"]).strip()
            
        # Clean physical index from aliases, keywords, synonyms
        for key in ("aliases", "keywords", "synonyms"):
            if node.get(key):
                node[key] = [strip_physical_index(item).strip() for item in node[key] if strip_physical_index(item).strip()]

        raw = (node.get("raw_content") or "").strip()
        if raw:
            node["raw_content"] = sanitize_raw_content(raw)
            if contains_garbage_artifact(node["raw_content"]):
                node["raw_content"] = ""
                node["retrieval_ready"] = False
                node["is_retrieval_chunk"] = False
                
        # Fix front matter page leak
        if node.get("page_start") == 0 and ntype == "CONTENT":
            m = re.search(r"<\s*physical_index_(\d+)\s*>", raw)
            if m:
                node["page_start"] = int(m.group(1))
                if node.get("page_end", 0) == 0:
                    node["page_end"] = node["page_start"]
            else:
                node["type"] = "FRONT_MATTER"
                node["is_front_matter"] = True
                node["retrieval_ready"] = False
                node["is_retrieval_chunk"] = False
                ntype = "FRONT_MATTER"

        if ntype in FRONT_MATTER_TYPES or ntype == "TABLE_OF_CONTENTS" or node.get("is_front_matter"):
            node["is_front_matter"] = True
            node["retrieval_ready"] = False
            node["is_retrieval_chunk"] = False

        if ntype in RETRIEVAL_TYPES and node.get("raw_content"):
            enrich_node(node, build_cfg)
        elif ntype in CONTAINER_TYPES:
            node["raw_content"] = ""
            node.pop("compressed_content", None)
            node.pop("token_count_compressed", None)

        for ch in node.get("nodes") or []:
            walk(ch, node)

    walk(root, None)
    rebuild_paths(root)
    finalize_children(root)
    
    # 2. Group chapters under DOCUMENT node
    doc_node = None
    def find_doc(n):
        nonlocal doc_node
        if n.get("type") == "DOCUMENT":
            doc_node = n
            return
        for c in n.get("nodes") or []:
            find_doc(c)
            
    find_doc(root)
    if doc_node:
        counter = [0]
        def count_nodes(n):
            for c in n.get("nodes") or []:
                count_nodes(c)
        count_nodes(root)
        group_chapters(doc_node, counter)
        
    # Rebuild paths and finalize children after grouping
    rebuild_paths(root)
    finalize_children(root)
    
    # 3. Apply micro_summary word caps and templates
    def walk_summary(node: dict) -> None:
        ntype = normalize_type(node.get("type", ""))
        title = node.get("title") or ""
        children_ids = node.get("children") or node.get("children_ids") or []
        
        if ntype == "TABLE_OF_CONTENTS":
            node["micro_summary"] = "Table of contents (navigation only)."
        elif ntype in FRONT_MATTER_TYPES or node.get("is_front_matter"):
            node["micro_summary"] = "Front matter (non-retrieval)."
        elif ntype in CONTAINER_TYPES:
            node["micro_summary"] = f"{title}: contains {len(children_ids)} subsections."
        else:
            # It's a CONTENT node
            micro = (node.get("micro_summary") or "").strip()
            if not micro:
                from .processing import micro_summary_from_content
                micro = micro_summary_from_content(title, node.get("raw_content") or "")
            
            # 1. Capped at 60 words first
            words = micro.split()
            if len(words) > 60:
                micro = " ".join(words[:60]).strip()
                
            # 2. Limit to at most 3 sentences
            sents = _split_micro_sentences(micro)
            if len(sents) > 3:
                sents = sents[:3]
                micro = " ".join(sents).strip()
            
            # 3. Ensure it ends with proper sentence punctuation
            if micro and micro[-1] not in ('.', '?', '!'):
                micro = micro.rstrip(' .!?') + "."
                
            node["micro_summary"] = micro
            
        for ch in node.get("nodes") or []:
            walk_summary(ch)
            
    walk_summary(root)
    
    # 4. Compute aggregate token counts
    compute_aggregate_tokens(root)
    
    # 5. Compute structural content hashes
    all_nodes = {}
    def build_node_index(n):
        nid = n.get("node_id")
        if nid:
            all_nodes[nid] = n
        for c in n.get("nodes") or []:
            build_node_index(c)
    build_node_index(root)
    compute_structural_content_hashes(root, all_nodes)

