import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading
import time
import json

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    logging.warning("Boto3 not available. Cloud features will be disabled.")

class UnifiedStorageManager:
    """Unified storage manager supporting both cloud and offline storage"""

    def __init__(self, bucket_name: str = None, region: str = "us-east-1", offline_storage_dir: str = "./offline_storage"):
        self.bucket_name = bucket_name or os.getenv("AWS_BUCKET_NAME")
        self.region = region
        self.offline_storage_dir = Path(offline_storage_dir)

        # Initialize cloud storage
        self.cloud_manager = CloudStorageManager(bucket_name, region)
        self.use_cloud = self.cloud_manager.is_available()

        # Initialize offline storage
        self.offline_manager = OfflineStorageManager(self.offline_storage_dir)

class CloudStorageManager:
    """Manages cloud storage operations for distributed audio processing"""

    def __init__(self, bucket_name: Optional[str] = None, region: str = "us-east-1"):
        self.bucket_name = bucket_name or os.getenv("AWS_BUCKET_NAME")
        self.region = region
        self.s3_client = None
        self._connection_pool = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize S3 client with connection pooling"""
        if not HAS_BOTO3:
            self.s3_client = None
            return

        try:
            # Import resource pool for connection management
            from resource_pool import get_connection

            # Use connection caching for S3 clients
            def create_s3_client():
                return boto3.client(
                    's3',
                    region_name=self.region
                )

            self.s3_client = get_connection(f"s3_{self.bucket_name}_{self.region}", create_s3_client)
            logging.info("S3 client initialized successfully with connection caching")
        except NoCredentialsError:
            logging.warning("AWS credentials not found. Cloud features will be disabled.")
            self.s3_client = None
        except Exception as e:
            logging.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None

    def is_available(self) -> bool:
        """Check if cloud storage is available"""
        return self.s3_client is not None and self.bucket_name is not None

    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """Upload a file to S3 with retry logic"""
        if not self.is_available() or self.s3_client is None:
            logging.warning("Cloud storage not available, upload skipped")
            return False

        if not os.path.exists(local_path):
            logging.error(f"Local file does not exist: {local_path}")
            return False

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
                logging.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
                return True
            except ClientError as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to upload {local_path} after {max_retries} attempts: {e}")
                    return False
                else:
                    logging.warning(f"Upload attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logging.error(f"Unexpected error during upload: {e}")
                return False

        return False

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download a file from S3 with retry logic"""
        if not self.is_available() or self.s3_client is None:
            logging.warning("Cloud storage not available, download skipped")
            return False

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.s3_client.download_file(self.bucket_name, s3_key, local_path)
                logging.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
                return True
            except ClientError as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to download {s3_key} after {max_retries} attempts: {e}")
                    return False
                else:
                    logging.warning(f"Download attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logging.error(f"Unexpected error during download: {e}")
                return False

        return False

    def list_files(self, prefix: str = "") -> List[str]:
        """List files in S3 bucket with optional prefix"""
        if not self.is_available() or self.s3_client is None:
            return []

        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            keys = []
            for page in pages:
                if 'Contents' in page:
                    keys.extend([obj['Key'] for obj in page['Contents']])
            return keys
        except ClientError as e:
            logging.error(f"Failed to list files: {e}")
            return []

    def delete_file(self, s3_key: str) -> bool:
        """Delete a file from S3"""
        if not self.is_available() or self.s3_client is None:
            return False

        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logging.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logging.error(f"Failed to delete {s3_key}: {e}")
            return False

    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """Generate a presigned URL for temporary access"""
        if not self.is_available() or self.s3_client is None:
            return None

        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logging.error(f"Failed to generate presigned URL for {s3_key}: {e}")
            return None

class DistributedProcessor:
    """Manages distributed audio processing across multiple nodes"""

    def __init__(self, cloud_manager: Optional[CloudStorageManager] = None):
        self.cloud_manager = cloud_manager or CloudStorageManager()
        self.workers: Dict[str, 'ProcessingNode'] = {}
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def register_worker(self, worker_id: str, capabilities: Dict[str, Any]):
        """Register a processing worker node"""
        with self.lock:
            self.workers[worker_id] = ProcessingNode(worker_id, capabilities, self.cloud_manager)
            logging.info(f"Registered worker {worker_id}")

    def unregister_worker(self, worker_id: str):
        """Unregister a processing worker node"""
        with self.lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                logging.info(f"Unregistered worker {worker_id}")
    def submit_task(self, task_data: Dict[str, Any]) -> str:
        """Submit a processing task for distributed execution"""
        import uuid
        task_id = f"task_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        task = {
            "task_id": task_id,
            "data": task_data,
            "status": "pending",
            "submitted_at": time.time(),
            "assigned_worker": None,
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None
        }

        with self.lock:
            self.tasks[task_id] = task

        # Try to assign task to a worker
        self._assign_task(task_id)

        logging.info(f"Submitted task {task_id}")
        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a distributed task"""
        with self.lock:
            return self.tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a distributed task"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task["status"] in ["pending", "running"]:
                    task["status"] = "cancelled"
                    task["completed_at"] = time.time()
                    return True
        return False

    def _assign_task(self, task_id: str):
        """Assign a task to an available worker"""
        with self.lock:
            if task_id not in self.tasks:
                return

            task = self.tasks[task_id]
            if task["status"] != "pending":
                return

            # Find available worker
            for worker_id, worker in self.workers.items():
                if worker.is_available():
                    task["assigned_worker"] = worker_id
                    task["status"] = "running"
                    task["started_at"] = time.time()

                    # Submit task to worker
                    threading.Thread(
                        target=self._execute_task_on_worker,
                        args=(task_id, worker),
                        daemon=True
                    ).start()
                    break

    def _execute_task_on_worker(self, task_id: str, worker: 'ProcessingNode'):
        """Execute a task on a specific worker"""
        try:
            with self.lock:
                if task_id not in self.tasks:
                    return
                task = self.tasks[task_id]

            # Execute the task
            result = worker.process_task(task["data"])

            with self.lock:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    task["status"] = "completed"
                    task["completed_at"] = time.time()
                    task["result"] = result

        except Exception as e:
            with self.lock:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    task["status"] = "failed"
                    task["completed_at"] = time.time()
                    task["error"] = str(e)

            logging.error(f"Task {task_id} failed on worker {worker.worker_id}: {e}")

class ProcessingNode:
    """Represents a processing node in the distributed system"""

    def __init__(self, worker_id: str, capabilities: Dict[str, Any], cloud_manager: CloudStorageManager):
        self.worker_id = worker_id
        self.capabilities = capabilities
        self.cloud_manager = cloud_manager
        self.current_task: Optional[Dict[str, Any]] = None
        self.lock = threading.Lock()

    def is_available(self) -> bool:
        """Check if this worker is available for new tasks"""
        with self.lock:
            return self.current_task is None

    def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task on this worker node"""
        with self.lock:
            self.current_task = task_data

        try:
            task_type = task_data.get("type", "compression")

            if task_type == "compression":
                return self._process_compression_task(task_data)
            elif task_type == "analysis":
                return self._process_analysis_task(task_data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

        finally:
            with self.lock:
                self.current_task = None

    def _process_compression_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an audio compression task"""
        input_key = task_data["input_key"]
        output_key = task_data["output_key"]
        compression_params = task_data["params"]

        # Download input file from cloud
        local_input = f"/tmp/{self.worker_id}_{os.path.basename(input_key)}"
        if not self.cloud_manager.download_file(input_key, local_input):
            raise Exception(f"Failed to download input file {input_key}")

        try:
            # Perform compression
            from compress_audio import compress_audio

            local_output = f"/tmp/{self.worker_id}_{os.path.basename(output_key)}"

            success, input_size, output_size = compress_audio(
                input_file=local_input,
                output_file=local_output,
                **compression_params
            )

            if not success:
                raise Exception("Compression failed")

            # Upload result to cloud
            if not self.cloud_manager.upload_file(local_output, output_key):
                raise Exception(f"Failed to upload output file {output_key}")

            # Clean up local files
            os.remove(local_input)
            os.remove(local_output)

            return {
                "success": True,
                "input_size": input_size,
                "output_size": output_size,
                "compression_ratio": (1 - output_size / input_size) * 100 if input_size > 0 else 0
            }

        except Exception as e:
            # Clean up on error
            if os.path.exists(local_input):
                os.remove(local_input)
            if 'local_output' in locals() and os.path.exists(local_output):
                os.remove(local_output)
            raise e

    def _process_analysis_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an audio analysis task"""
        input_key = task_data["input_key"]

        # Download input file from cloud
        local_input = f"/tmp/{self.worker_id}_{os.path.basename(input_key)}"
        if not self.cloud_manager.download_file(input_key, local_input):
            raise Exception(f"Failed to download input file {input_key}")

        try:
            # Perform analysis - lazy load audio_analyzer
            from compress_audio import _get_audio_analyzer
            audio_analyzer = _get_audio_analyzer()

            analysis = audio_analyzer.analyze_file(local_input)

            # Clean up
            os.remove(local_input)

            return {
                "success": True,
                "analysis": analysis
            }

        except Exception as e:
            # Clean up on error
            if os.path.exists(local_input):
                os.remove(local_input)
            raise e

class OfflineStorageManager:
    """Manages offline/local storage operations"""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "metadata.json"
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata about stored files"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save metadata about stored files"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except IOError:
            logging.warning("Could not save offline storage metadata")

    def store_file(self, local_path: str, key: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a file in offline storage"""
        try:
            # Create subdirectories if needed
            dest_path = self.storage_dir / key
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            import shutil
            shutil.copy2(local_path, dest_path)

            # Update metadata
            file_info = {
                "original_path": local_path,
                "stored_at": time.time(),
                "size": os.path.getsize(local_path),
                "key": key
            }
            if metadata:
                file_info.update(metadata)

            self.metadata[key] = file_info
            self._save_metadata()

            logging.info(f"Stored file offline: {key}")
            return True

        except Exception as e:
            logging.error(f"Failed to store file offline {key}: {e}")
            return False

    def retrieve_file(self, key: str, local_path: str) -> bool:
        """Retrieve a file from offline storage"""
        try:
            source_path = self.storage_dir / key
            if not source_path.exists():
                logging.error(f"File not found in offline storage: {key}")
                return False

            # Copy file
            import shutil
            shutil.copy2(source_path, local_path)

            logging.info(f"Retrieved file from offline storage: {key}")
            return True

        except Exception as e:
            logging.error(f"Failed to retrieve file from offline storage {key}: {e}")
            return False

    def list_files(self, prefix: str = "") -> List[str]:
        """List files in offline storage with optional prefix"""
        try:
            files = []
            for key in self.metadata.keys():
                if key.startswith(prefix):
                    files.append(key)
            return files
        except Exception:
            return []

    def delete_file(self, key: str) -> bool:
        """Delete a file from offline storage"""
        try:
            file_path = self.storage_dir / key
            if file_path.exists():
                os.remove(file_path)

            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()

            logging.info(f"Deleted file from offline storage: {key}")
            return True

        except Exception as e:
            logging.error(f"Failed to delete file from offline storage {key}: {e}")
            return False

    def get_file_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get information about a stored file"""
        return self.metadata.get(key)

    def cleanup_old_files(self, max_age_days: int = 30) -> int:
        """Clean up files older than specified days"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            deleted_count = 0

            keys_to_delete = []
            for key, info in self.metadata.items():
                if current_time - info.get("stored_at", 0) > max_age_seconds:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                if self.delete_file(key):
                    deleted_count += 1

            if deleted_count > 0:
                logging.info(f"Cleaned up {deleted_count} old files from offline storage")

            return deleted_count

        except Exception as e:
            logging.error(f"Failed to cleanup old files: {e}")
            return 0

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            total_size = 0
            file_count = 0

            for info in self.metadata.values():
                total_size += info.get("size", 0)
                file_count += 1

            return {
                "total_files": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "storage_path": str(self.storage_dir)
            }

        except Exception:
            return {"error": "Could not get storage stats"}

# Unified storage manager
class StorageManager:
    """Unified storage manager supporting both cloud and offline storage"""

    def __init__(self, bucket_name: Optional[str] = None, region: str = "us-east-1", offline_storage_dir: str = "./offline_storage"):
        self.bucket_name = bucket_name or os.getenv("AWS_BUCKET_NAME")
        self.region = region
        self.offline_storage_dir = Path(offline_storage_dir)

        # Initialize cloud storage
        self.cloud_manager = CloudStorageManager(bucket_name, region)
        self.use_cloud = self.cloud_manager.is_available()

        # Initialize offline storage
        self.offline_manager = OfflineStorageManager(self.offline_storage_dir)

    def store_file(self, local_path: str, key: str, metadata: Dict[str, Any] = None, prefer_cloud: bool = True) -> bool:
        """Store a file using preferred storage method with graceful degradation"""
        if not os.path.exists(local_path):
            logging.error(f"Local file does not exist: {local_path}")
            return False

        if prefer_cloud and self.use_cloud:
            try:
                success = self.cloud_manager.upload_file(local_path, key)
                if success:
                    return True
                else:
                    logging.warning("Cloud upload failed, falling back to offline storage")
            except Exception as e:
                logging.warning(f"Cloud storage error, falling back to offline storage: {e}")

        # Fallback to offline storage
        try:
            return self.offline_manager.store_file(local_path, key, metadata)
        except Exception as e:
            logging.error(f"Failed to store file in offline storage: {e}")
            return False

    def retrieve_file(self, key: str, local_path: str, prefer_cloud: bool = True) -> bool:
        """Retrieve a file using preferred storage method"""
        if prefer_cloud and self.use_cloud:
            success = self.cloud_manager.download_file(key, local_path)
            if success:
                return True

        # Fallback to offline storage
        return self.offline_manager.retrieve_file(key, local_path)

    def list_files(self, prefix: str = "", include_offline: bool = True) -> List[str]:
        """List files from all storage methods"""
        files = []

        if self.use_cloud:
            files.extend(self.cloud_manager.list_files(prefix))

        if include_offline:
            files.extend(self.offline_manager.list_files(prefix))

        return files

    def delete_file(self, key: str) -> bool:
        """Delete a file from all storage methods"""
        cloud_success = True
        offline_success = True

        if self.use_cloud:
            cloud_success = self.cloud_manager.delete_file(key)

        offline_success = self.offline_manager.delete_file(key)

        return cloud_success and offline_success

    def get_file_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get file information, preferring cloud if available"""
        # Cloud doesn't have detailed metadata, so check offline
        return self.offline_manager.get_file_info(key)

    def cleanup_storage(self, max_age_days: int = 30) -> Dict[str, int]:
        """Clean up old files from storage"""
        results = {
            "offline_cleaned": self.offline_manager.cleanup_old_files(max_age_days),
            "cloud_cleaned": 0  # Cloud cleanup not implemented
        }
        return results

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        stats = {
            "cloud_available": self.use_cloud,
            "offline": self.offline_manager.get_storage_stats()
        }

        if self.use_cloud:
            stats["cloud"] = {
                "bucket": self.bucket_name,
                "region": self.region
            }

        return stats

# Global instances (lazy initialization)
_cloud_manager = None
_storage_manager = None
_distributed_processor = None

# Lock for thread-safe lazy initialization
_init_lock = threading.Lock()

def get_cloud_manager() -> CloudStorageManager:
    global _cloud_manager
    if _cloud_manager is None:
        with _init_lock:
            if _cloud_manager is None:
                _cloud_manager = CloudStorageManager()
    return _cloud_manager

def get_storage_manager() -> StorageManager:
    global _storage_manager
    if _storage_manager is None:
        with _init_lock:
            if _storage_manager is None:
                _storage_manager = StorageManager()
    return _storage_manager

def get_distributed_processor() -> DistributedProcessor:
    global _distributed_processor
    if _distributed_processor is None:
        with _init_lock:
            if _distributed_processor is None:
                _distributed_processor = DistributedProcessor(get_cloud_manager())
    return _distributed_processor