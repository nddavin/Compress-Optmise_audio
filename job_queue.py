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
    preset_name: Optional[str] = None  # Name of the preset used
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
            "preset_name": self.preset_name,
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
            preset_name=data.get("preset_name"),
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
            self.queue.put((-job.priority.value, job.created_at, job.job_id))
            self._save_jobs()

        logging.info(f"Added job {job.job_id} to queue (preset: {job.preset_name or 'none'})")
        return job.job_id

    def add_preset_batch(self, input_dir: str, output_dir: str, preset_name: str) -> List[str]:
        """Add a batch of jobs using a preset configuration"""
        from presets import preset_manager

        preset = preset_manager.get_preset(preset_name)
        if not preset:
            raise ValueError(f"Preset '{preset_name}' not found")

        # Build filter chain for the preset
        from compress_audio import build_audio_filters
        filter_chain = build_audio_filters(
            loudnorm_enabled=preset.loudnorm_enabled,
            compressor_enabled=preset.compressor_enabled,
            compressor_preset=preset.compressor_preset,
            multiband_enabled=preset.multiband_enabled,
            multiband_preset=preset.multiband_preset,
            ml_noise_reduction=preset.ml_noise_reduction,
            silence_trim_enabled=preset.silence_trim_enabled,
            noise_gate_enabled=preset.noise_gate_enabled,
            channels=preset.channels
        )

        # Create output directories
        from compress_audio import create_output_dirs
        output_dirs = create_output_dirs(output_dir, preset.bitrates)

        # Find audio files
        import os
        extensions = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus")
        audio_files = []

        for file in os.listdir(input_dir):
            if file.lower().endswith(extensions):
                audio_files.append(os.path.join(input_dir, file))

        if not audio_files:
            raise ValueError(f"No audio files found in {input_dir}")

        # Create jobs for each file and bitrate combination
        job_ids = []
        for audio_file in audio_files:
            filename = os.path.splitext(os.path.basename(audio_file))[0]

            for bitrate in preset.bitrates:
                from compress_audio import get_format_defaults
                format_info = get_format_defaults(preset.format)
                ext = format_info.get("ext", ".mp3")
                output_file = os.path.join(output_dirs[bitrate], f"{filename}{ext}")

                job = CompressionJob(
                    job_id=f"{preset_name}_{filename}_{bitrate}",
                    input_file=audio_file,
                    output_file=output_file,
                    bitrate=bitrate,
                    format=preset.format,
                    filter_chain=filter_chain,
                    channels=preset.channels,
                    preserve_metadata=True,
                    preset_name=preset_name
                )

                job_ids.append(self.add_job(job))

        logging.info(f"Added {len(job_ids)} jobs for preset '{preset_name}'")
        return job_ids

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
        """Process a single compression job with comprehensive error recovery"""
        try:
            from compress_audio import compress_audio

            # Update progress
            job.progress = 10.0
            job.updated_at = time.time()

            # Validate job inputs before processing
            if not os.path.exists(job.input_file):
                job.error_message = f"Input file does not exist: {job.input_file}"
                logging.error(job.error_message)
                return False

            if not os.access(job.input_file, os.R_OK):
                job.error_message = f"Input file is not readable: {job.input_file}"
                logging.error(job.error_message)
                return False

            # Ensure output directory exists
            output_dir = os.path.dirname(job.output_file)
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except OSError as e:
                    job.error_message = f"Cannot create output directory: {e}"
                    logging.error(job.error_message)
                    return False

            job.progress = 25.0
            job.updated_at = time.time()

            # Perform compression with retry logic
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
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

                    if success:
                        job.progress = 100.0
                        job.input_size = input_size
                        job.output_size = output_size
                        job.updated_at = time.time()
                        return True
                    else:
                        if attempt == max_retries:
                            job.error_message = "Compression failed after all retries"
                            logging.error(f"Job {job.job_id} failed: {job.error_message}")
                            return False
                        else:
                            logging.warning(f"Compression attempt {attempt + 1} failed for job {job.job_id}, retrying")
                            time.sleep(1)

                except Exception as e:
                    if attempt == max_retries:
                        job.error_message = f"Compression failed: {str(e)}"
                        logging.error(f"Job {job.job_id} failed after {max_retries + 1} attempts: {e}")
                        return False
                    else:
                        logging.warning(f"Compression attempt {attempt + 1} failed for job {job.job_id}: {e}")
                        time.sleep(1)

            job.error_message = "Compression failed after all retries"
            return False

        except Exception as e:
            job.error_message = f"Unexpected error: {str(e)}"
            logging.error(f"Job {job.job_id} failed with unexpected error: {e}")
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