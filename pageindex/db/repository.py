from sqlalchemy.orm import Session
from sqlalchemy import func
from .models import DocumentJob, DocumentNode
from .node_order import ordered_flat_nodes
from .sort_ids import sortable_node_row_id

class IngestionRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_job(self, doc_id: str) -> DocumentJob | None:
        return self.db.query(DocumentJob).filter(DocumentJob.doc_id == doc_id).first()

    def upsert_job(
        self,
        doc_id: str,
        status: str,
        error_message: str = None,
        results: dict = None,
        *,
        file_name: str | None = None,
    ) -> DocumentJob:
        job = self.get_job(doc_id)
        if not job:
            next_seq = (self.db.query(func.max(DocumentJob.seq_id)).scalar() or 0) + 1
            job = DocumentJob(
                doc_id=doc_id,
                seq_id=next_seq,
                file_name=file_name,
                status=status,
                error_message=error_message,
                results=results,
            )
            self.db.add(job)
        else:
            job.status = status
            if file_name is not None:
                job.file_name = file_name
            if job.seq_id is None:
                job.seq_id = (self.db.query(func.max(DocumentJob.seq_id)).scalar() or 0) + 1
            if error_message is not None:
                job.error_message = error_message
            if results is not None:
                job.results = results
        self.db.commit()
        self.db.refresh(job)
        return job

    def replace_nodes(self, doc_id: str, flat_nodes: list[dict]):
        job = self.get_job(doc_id)
        self.db.query(DocumentNode).filter(DocumentNode.doc_id == doc_id).delete()

        db_nodes = []
        id_map: dict[str, str] = {}
        ordered = ordered_flat_nodes(flat_nodes)
        for i, n in enumerate(ordered, start=1):
            old = str(n.get("node_id") or "")
            id_map[old] = f"{i:04d}"

        for n in ordered:
            metadata = {
                "page_start": n.get("page_start") or n.get("page_index"),
                "page_end": n.get("page_end"),
                "char_start": n.get("char_start"),
                "char_end": n.get("char_end"),
                "aliases": n.get("aliases"),
                "keywords": n.get("keywords"),
                "synonyms": n.get("synonyms"),
            }

            old_node_id = str(n.get("node_id") or "")
            old_parent_id = n.get("parent_id")
            old_parent_id_str = str(old_parent_id) if old_parent_id is not None else ""

            new_node_id = id_map.get(old_node_id) or "0000"
            new_parent_id = id_map.get(old_parent_id_str) if old_parent_id is not None else None

            node_json = dict(n)
            node_json.setdefault("_original_node_id", old_node_id)
            if old_parent_id is not None:
                node_json.setdefault("_original_parent_id", old_parent_id_str)

            job_seq = getattr(job, "seq_id", None) if job else None
            db_nodes.append(
                DocumentNode(
                    id=sortable_node_row_id(job_seq, new_node_id),
                    doc_id=doc_id,
                    seq_id=job_seq,
                    file_name=getattr(job, "file_name", None) if job else None,
                    node_id=new_node_id,
                    parent_id=new_parent_id,
                    type=n.get("type") or "unknown",
                    title=n.get("title") or "Untitled",
                    path=n.get("path"),
                    level=n.get("level", 0),
                    raw_content=n.get("raw_content") or n.get("text"),
                    compressed_content=n.get("compressed_content"),
                    micro_summary=n.get("micro_summary") or n.get("summary"),
                    content_hash=n.get("content_hash"),
                    retrieval_ready=n.get("retrieval_ready", False),
                    is_front_matter=n.get("is_front_matter", False),
                    metadata_json=metadata,
                    node_json=node_json,
                )
            )

        self.db.bulk_save_objects(db_nodes)
        self.db.commit()
