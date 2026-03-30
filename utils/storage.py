import os
import logging
from pathlib import Path
from typing import Optional
from configs.settings import settings

logger = logging.getLogger(__name__)

class CloudStorage:
    """
    Utility for managing PDF file storage on cloud (Supabase) or locally.
    """

    def __init__(self):
        self.supabase_url = settings.SUPABASE_URL
        self.supabase_key = settings.SUPABASE_KEY
        self.bucket = settings.SUPABASE_BUCKET
        self.client = None

        if self.supabase_url and self.supabase_key:
            try:
                from supabase import create_client
                self.client = create_client(self.supabase_url, self.supabase_key)
                logger.info("Supabase Storage initialized.")
            except ImportError:
                logger.warning("Supabase client not installed. Falling back to local storage.")
            except Exception as e:
                logger.error("Failed to Initialize Supabase Client: %s", e)

    def upload_file(self, local_path: Path, remote_name: str) -> Optional[str]:
        """
        Uploads a file to Supabase Storage if configured, otherwise remains local.
        Returns the remote URL or local path string.
        """
        if self.client:
            try:
                with open(local_path, "rb") as f:
                    # Upsert (overwrite if exists)
                    self.client.storage.from_(self.bucket).upload(
                        path=remote_name, 
                        file=f,
                        file_options={"upsert": "true"}
                    )
                # Generate a public or signed URL if needed
                # For now just return the path in bucket
                logger.info("Successfully uploaded %s to Supabase.", remote_name)
                return f"supabase://{self.bucket}/{remote_name}"
            except Exception as e:
                logger.error("Supabase Upload Error: %s", e)
                return None
        
        logger.info("Supabase not configured, file stored locally at %s", local_path)
        return str(local_path)

    def download_file(self, remote_name: str, local_dest: Path) -> bool:
        """
        Downloads a file from Supabase Storage to local destination.
        """
        if self.client:
            try:
                res = self.client.storage.from_(self.bucket).download(remote_name)
                local_dest.write_bytes(res)
                logger.info("Successfully downloaded %s from Supabase.", remote_name)
                return True
            except Exception as e:
                logger.error("Supabase Download Error: %s", e)
                return False
        
        # If not cloud, assume the local_dest already has the file or it's irrelevant
        return local_dest.exists()

storage_client = CloudStorage()
