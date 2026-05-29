import asyncio
import os
import uuid
from pathlib import Path
from fastapi import BackgroundTasks
from pageindex.log_util import log_info, log_error, log_exception
from pageindex.db.database import SessionLocal
from pageindex.db.repository import IngestionRepository
from pageindex.s3_reader import list_doc_ids, get_pdf_key, download_s3_to_tempfile
from pageindex.env_settings import settings
from pageindex.cloud import build_cloud_index
from pageindex.usage.meter import UsageMeter

class IngestionService:
    def __init__(self):
        self._is_running = False

    async def run(self):
        if self._is_running:
            log_info("Ingestion batch already running.")
            return
        
        self._is_running = True
        log_info("Starting batch ingestion from S3...")
        try:
            doc_ids = list_doc_ids()
            if not doc_ids:
                log_info("No document IDs found in S3 under prefix %s.", settings.S3_CLEANED_PREFIX)
                return

            log_info("Found %d docs to process.", len(doc_ids))
            
            semaphore = asyncio.Semaphore(settings.INGESTION_MAX_CONCURRENT_DOCS)
            tasks = [self.process_document_with_semaphore(doc_id, semaphore) for doc_id in doc_ids]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            log_info("Batch ingestion completed.")
        except Exception as e:
            log_exception("Batch ingestion failed: %s", e)
        finally:
            self._is_running = False

    async def process_document_with_semaphore(self, doc_id: str, semaphore: asyncio.Semaphore):
        async with semaphore:
            await self.process_document(doc_id)

    async def process_document(self, doc_id: str):
        db = SessionLocal()
        repo = IngestionRepository(db)
        try:
            job = repo.get_job(doc_id)
            if job and job.status == "completed":
                log_info("Doc %s already completed, skipping.", doc_id)
                return

            repo.upsert_job(doc_id, status="processing")
            log_info("Processing doc_id: %s", doc_id)

            # Locate PDF
            pdf_key = get_pdf_key(doc_id)
            if not pdf_key:
                repo.upsert_job(doc_id, status="failed", error_message="PDF missing in S3")
                log_error("Doc %s: PDF missing in S3.", doc_id)
                return
            file_name = Path(pdf_key).name
            repo.upsert_job(doc_id, status="processing", file_name=file_name)

            # Download
            temp_path = download_s3_to_tempfile(pdf_key)
            
            # Process via ThreadPool since build_cloud_index is sync and heavy
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self._run_pipeline, temp_path, doc_id)

            # Flatten nodes
            structure = result.get("structure_vrag") or result.get("structure")
            if structure:
                tree_nodes = structure if isinstance(structure, list) else structure.get("nodes", [])
                
                flat_nodes = []
                def flatten(nodes):
                    for n in nodes:
                        flat_nodes.append(n)
                        flatten(n.get("nodes", []))
                flatten(tree_nodes)
                
                repo.replace_nodes(doc_id, flat_nodes)
                
            repo.upsert_job(doc_id, status="completed", results=result, file_name=file_name)
            log_info("Doc %s successfully ingested.", doc_id)

        except Exception as e:
            from pageindex.api_errors import format_user_error

            log_exception("Error processing doc_id %s: %s", doc_id, e)
            db.rollback()  # Rollback any failed transactions so upsert_job doesn't fail with PendingRollbackError
            fn = locals().get("file_name")
            repo.upsert_job(doc_id, status="failed", error_message=format_user_error(e)[:4000], file_name=fn)
        finally:
            db.close()
            # Cleanup temp file
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _run_pipeline(self, pdf_path: str, doc_id: str) -> dict:
        fresh_job_id = uuid.uuid4().hex
        meter = UsageMeter(
            job_id=fresh_job_id,
            document_id=doc_id,
            document_name=Path(pdf_path).name,
            pipeline="pageindex",
        )
        return build_cloud_index(pdf_path, meter=meter, job_id=fresh_job_id)

ingestion_service = IngestionService()
