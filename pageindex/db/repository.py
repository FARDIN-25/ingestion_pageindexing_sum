import uuid
from sqlalchemy.orm import Session
from .models import DocumentJob, DocumentNode

class IngestionRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_job(self, doc_id: str) -> DocumentJob | None:
        return self.db.query(DocumentJob).filter(DocumentJob.doc_id == doc_id).first()

    def upsert_job(self, doc_id: str, status: str, error_message: str = None, results: dict = None) -> DocumentJob:
        job = self.get_job(doc_id)
        if not job:
            job = DocumentJob(doc_id=doc_id, status=status, error_message=error_message, results=results)
            self.db.add(job)
        else:
            job.status = status
            if error_message is not None:
                job.error_message = error_message
            if results is not None:
                job.results = results
        self.db.commit()
        self.db.refresh(job)
        return job

    def replace_nodes(self, doc_id: str, flat_nodes: list[dict]):
        self.db.query(DocumentNode).filter(DocumentNode.doc_id == doc_id).delete()
        
        db_nodes = []
        for n in flat_nodes:
            metadata = {
                "page_start": n.get("page_start") or n.get("page_index"),
                "page_end": n.get("page_end"),
                "char_start": n.get("char_start"),
                "char_end": n.get("char_end"),
                "aliases": n.get("aliases"),
                "keywords": n.get("keywords"),
                "synonyms": n.get("synonyms")
            }
            
            db_nodes.append(
                DocumentNode(
                    id=uuid.uuid4().hex,
                    doc_id=doc_id,
                    node_id=n.get("node_id"),
                    parent_id=n.get("parent_id"),
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
                    node_json=n
                )
            )
        
        self.db.bulk_save_objects(db_nodes)
        self.db.commit()
