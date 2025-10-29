import threading
import queue
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class CompressionJob:
    """Represents a single audio compression job"""
    job_id: str
    input_file: str
    output_file: str
    bitrate: int
    format: str
    filter_chain: Optional[str] = None
    channels: int = 1
    preserve_metadata: bool = True
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    error_message: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    input_size: int = 0
    output_size: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "bitrate": self.bitrate,
            "format": self.format,
            "filter_chain": self.filter_chain,
            "channels": self.channels,
            "preserve_metadata": self.preserve_metadata,
            "priority": self.priority.value,
            "status": self.status.value,
            "progress": self.progress,
            "error_message": self.error_message,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompressionJob':
        return cls(
            job_id=data["job_id"],
            input_file=data["input_file"],
            output_file=data["output_file"],
            bitrate=data["bitrate"],
            format=data["format"],
            filter_chain=data.get("filter_chain"),
            channels=data.get("channels", 1),
            preserve_metadata=data.get("preserve_metadata", True),
            priority=JobPriority(data.get("priority", JobPriority.NORMAL.value)),
            status=JobStatus(data["status"]),
            progress=data.get("progress", 0.0),
            error_message=data.get("error_message"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            input_size=data.get("input_size", 0),
            output_size=data.get("output_size", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time())
        )

class JobQueue:
    """Thread-safe job queue for batch audio compression"""

    def __init__(self, max_workers: int = 4, persist_file: str = "job_queue.json"):
        self.queue = queue.PriorityQueue()
        self.jobs: Dict[str, CompressionJob] = {}
        self.max_workers = max_workers
        self.workers: List[threading.Thread] = []
        self.running = False
        self.persist_file = Path(persist_file)
        self.lock = threading.Lock()
        self.callbacks: Dict[str, Callable] = {}

        # Load persisted jobs
        self._load_jobs()

    def start(self):
        """Start the job queue processing"""
        if self.running:
            return

        self.running = True
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"JobWorker-{i+1}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

        logging.info(f"Job queue started with {self.max_workers} workers")

    def stop(self):
        """Stop the job queue processing"""
        self.running = False
        self._save_jobs()

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)

        logging.info("Job queue stopped")

    def add_job(self, job: CompressionJob) -> str:
        """Add a job to the queue"""
        with self.lock:
            self.jobs[job.job_id] = job
            # Priority queue: (priority, created_at, job_id)
            self.queue.put((job.priority.value, job.created_at, job.job_id))
            self._save_jobs()

        logging.info(f"Added job {job.job_id} to queue")
        return job.job_id

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job"""
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job.status == JobStatus.PENDING:
                    job.status = JobStatus.CANCELLED
                    job.updated_at = time.time()
                    self._save_jobs()
                    return True
        return False

    def get_job_status(self, job_id: str) -> Optional[CompressionJob]:
        """Get the status of a job"""
        with self.lock:
            return self.jobs.get(job_id)

    def get_all_jobs(self) -> List[CompressionJob]:
        """Get all jobs"""
        with self.lock:
            return list(self.jobs.values())

    def get_pending_jobs(self) -> List[CompressionJob]:
        """Get pending jobs"""
        with self.lock:
            return [job for job in self.jobs.values() if job.status == JobStatus.PENDING]

    def get_running_jobs(self) -> List[CompressionJob]:
        """Get running jobs"""
        with self.lock:
            return [job for job in self.jobs.values() if job.status == JobStatus.RUNNING]

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for job events"""
        self.callbacks[event] = callback

    def _trigger_callback(self, event: str, job: CompressionJob):
        """Trigger a callback if registered"""
        if event in self.callbacks:
            try:
                self.callbacks[event](job)
            except Exception as e:
                logging.error(f"Error in callback for event {event}: {e}")

    def _worker_loop(self):
        """Main worker loop"""
        while self.running:
            try:
                # Get next job from queue
                priority, created_at, job_id = self.queue.get(timeout=1)

                with self.lock:
                    if job_id not in self.jobs:
                        continue

                    job = self.jobs[job_id]
                    if job.status != JobStatus.PENDING:
                        continue

                    # Mark job as running
                    job.status = JobStatus.RUNNING
                    job.start_time = time.time()
                    job.updated_at = time.time()

                self._trigger_callback("job_started", job)

                # Process the job
                success = self._process_job(job)

                with self.lock:
                    job.end_time = time.time()
                    job.updated_at = time.time()

                    if success:
                        job.status = JobStatus.COMPLETED
                        self._trigger_callback("job_completed", job)
                    else:
                        job.status = JobStatus.FAILED
                        self._trigger_callback("job_failed", job)

                self._save_jobs()

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in worker loop: {e}")

    def _process_job(self, job: CompressionJob) -> bool:
        """Process a single compression job"""
        try:
            from compress_audio import compress_audio

            # Update progress
            job.progress = 10.0
            job.updated_at = time.time()

            # Perform compression
            success, input_size, output_size = compress_audio(
                input_file=job.input_file,
                output_file=job.output_file,
                bitrate=job.bitrate,
                filter_chain=job.filter_chain,
                output_format=job.format,
                channels=job.channels,
                preserve_metadata=job.preserve_metadata,
                dry_run=False,
                preview_mode=False
            )

            job.progress = 100.0
            job.input_size = input_size
            job.output_size = output_size

            if not success:
                job.error_message = "Compression failed"
                return False

            return True

        except Exception as e:
            job.error_message = str(e)
            logging.error(f"Job {job.job_id} failed: {e}")
            return False

    def _save_jobs(self):
        """Save jobs to persistent storage"""
        try:
            jobs_data = {job_id: job.to_dict() for job_id, job in self.jobs.items()}
            with open(self.persist_file, 'w') as f:
                json.dump(jobs_data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save jobs: {e}")

    def _load_jobs(self):
        """Load jobs from persistent storage"""
        if not self.persist_file.exists():
            return

        try:
            with open(self.persist_file, 'r') as f:
                jobs_data = json.load(f)

            for job_id, job_data in jobs_data.items():
                job = CompressionJob.from_dict(job_data)
                self.jobs[job_id] = job

                # Re-queue pending jobs
                if job.status == JobStatus.PENDING:
                    self.queue.put((job.priority.value, job.created_at, job_id))

            logging.info(f"Loaded {len(self.jobs)} jobs from persistent storage")

        except Exception as e:
            logging.error(f"Failed to load jobs: {e}")

    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        with self.lock:
            stats = {
                "total": len(self.jobs),
                "pending": len([j for j in self.jobs.values() if j.status == JobStatus.PENDING]),
                "running": len([j for j in self.jobs.values() if j.status == JobStatus.RUNNING]),
                "completed": len([j for j in self.jobs.values() if j.status == JobStatus.COMPLETED]),
                "failed": len([j for j in self.jobs.values() if j.status == JobStatus.FAILED]),
                "cancelled": len([j for j in self.jobs.values() if j.status == JobStatus.CANCELLED])
            }
        return stats

# Global job queue instance
job_queue = JobQueue()