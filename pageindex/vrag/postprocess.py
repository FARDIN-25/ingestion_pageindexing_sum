"""Final tree passes: sanitize content, enrich metadata, ensure traversal fields."""

from __future__ import annotations



from typing import Any



from .processing import contains_garbage_artifact, sanitize_raw_content

from .schema import (

    CONTAINER_TYPES,

    FRONT_MATTER_TYPES,

    RETRIEVAL_TYPES,

    BuildConfig,

    enrich_node,

    finalize_children,

    normalize_type,

)

from .path_utils import rebuild_paths





def postprocess_vrag_tree(root: dict[str, Any], cfg: BuildConfig | None = None) -> None:

    """In-place cleanup and metadata completion on built or normalized trees."""

    build_cfg = cfg or BuildConfig()



    def walk(node: dict, parent: dict | None) -> None:

        ntype = normalize_type(node.get("type", ""))

        if parent:

            node["parent_id"] = parent.get("node_id")

        raw = (node.get("raw_content") or "").strip()

        if raw:

            node["raw_content"] = sanitize_raw_content(raw)

            if contains_garbage_artifact(node["raw_content"]):

                node["raw_content"] = ""

                node["retrieval_ready"] = False

                node["is_retrieval_chunk"] = False



        if ntype in FRONT_MATTER_TYPES or ntype == "TABLE_OF_CONTENTS":

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

