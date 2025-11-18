"""
Cloud Integration and API Backend for Pure Sound

This module provides enterprise-grade cloud integration including:
- RESTful and gRPC API endpoints for automation/CI integration
- Cloud sync with secure credential management
- Scalable distributed processing architecture
- Auto-scaling nodes with intelligent load balancing
- Docker containerization and orchestration support
"""

import asyncio
import logging
import json
import time
import uuid
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue

# FastAPI for REST endpoints
try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available, REST API disabled")

# gRPC support
try:
    import grpc
    from grpc import aio as grpc_aio
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    logging.warning("gRPC not available")

# AWS/S3 integration
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logging.warning("boto3 not available, cloud storage disabled")

# Docker/orchestration
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

from interfaces import IEventPublisher, IServiceProvider
from di_container import get_service
from security import security_manager, Permission, AuthMethod
from audio_analysis_enhanced import audio_analysis_engine, AudioAnalysisResult
from audio_processing_enhanced import audio_processing_engine, ProcessingJob


class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    LOCAL = "local"


class JobStatus(Enum):
    """API job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CloudConfig:
    """Cloud configuration settings"""
    provider: CloudProvider
    region: str = "us-east-1"
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    bucket_name: Optional[str] = None
    endpoint_url: Optional[str] = None
    verify_ssl: bool = True
    timeout: int = 300


@dataclass
class APIJob:
    """API job definition"""
    job_id: str
    input_file: str
    output_files: List[str]
    preset: str
    quality: str
    status: JobStatus
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    result_url: Optional[str] = None


class CloudStorageManager:
    """Manages cloud storage operations"""

    def __init__(self, config: CloudConfig):
        self.config = config
        self.s3_client = None
        self._init_client()

    def _init_client(self):
        """Initialize cloud storage client"""
        if not BOTO3_AVAILABLE:
            logging.error("boto3 not available, cloud storage disabled")
            return
        
        try:
            if self.config.provider == CloudProvider.AWS:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.config.access_key,
                    aws_secret_access_key=self.config.secret_key,
                    region_name=self.config.region,
                    endpoint_url=self.config.endpoint_url,
                    verify=self.config.verify_ssl
                )
            logging.info(f"Cloud storage client initialized for {self.config.provider.value}")
        except Exception as e:
            logging.error(f"Failed to initialize cloud storage client: {e}")
            self.s3_client = None

    def upload_file(self, local_path: str, cloud_path: str, 
                   metadata: Optional[Dict[str, str]] = None) -> bool:
        """Upload file to cloud storage"""
        if not self.s3_client or not self.config.bucket_name:
            return False
        
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.s3_client.upload_file(
                local_path, 
                self.config.bucket_name, 
                cloud_path,
                ExtraArgs=extra_args
            )
            logging.info(f"Uploaded {local_path} to {cloud_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to upload file: {e}")
            return False

    def download_file(self, cloud_path: str, local_path: str) -> bool:
        """Download file from cloud storage"""
        if not self.s3_client or not self.config.bucket_name:
            return False
        
        try:
            self.s3_client.download_file(
                self.config.bucket_name,
                cloud_path,
                local_path
            )
            logging.info(f"Downloaded {cloud_path} to {local_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to download file: {e}")
            return False

    def delete_file(self, cloud_path: str) -> bool:
        """Delete file from cloud storage"""
        if not self.s3_client or not self.config.bucket_name:
            return False
        
        try:
            self.s3_client.delete_object(
                Bucket=self.config.bucket_name,
                Key=cloud_path
            )
            logging.info(f"Deleted {cloud_path} from cloud storage")
            return True
        except Exception as e:
            logging.error(f"Failed to delete file: {e}")
            return False

    def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List files in cloud storage"""
        if not self.s3_client or not self.config.bucket_name:
            return []
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=prefix
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'etag': obj['ETag']
                    })
            
            return files
        except Exception as e:
            logging.error(f"Failed to list files: {e}")
            return []


class DistributedProcessingManager:
    """Manages distributed processing across multiple nodes"""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.active_jobs: Dict[str, str] = {}  # job_id -> node_id
        self.job_queue: List[str] = []
        self.node_lock = threading.RLock()

    def register_node(self, node_id: str, node_info: Dict[str, Any]) -> bool:
        """Register a processing node"""
        with self.node_lock:
            self.nodes[node_id] = {
                **node_info,
                'registered_at': time.time(),
                'last_heartbeat': time.time(),
                'active_jobs': 0,
                'status': 'active'
            }
            logging.info(f"Registered processing node: {node_id}")
            return True

    def unregister_node(self, node_id: str) -> bool:
        """Unregister a processing node"""
        with self.node_lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logging.info(f"Unregistered processing node: {node_id}")
                return True
            return False

    def heartbeat(self, node_id: str) -> bool:
        """Process node heartbeat"""
        with self.node_lock:
            if node_id in self.nodes:
                self.nodes[node_id]['last_heartbeat'] = time.time()
                return True
            return False

    def get_healthy_nodes(self) -> List[str]:
        """Get list of healthy processing nodes"""
        current_time = time.time()
        healthy_nodes = []
        
        with self.node_lock:
            for node_id, node_info in self.nodes.items():
                if current_time - node_info['last_heartbeat'] < 300:  # 5 minutes
                    if node_info['status'] == 'active':
                        healthy_nodes.append(node_id)
        
        return healthy_nodes

    def get_least_loaded_node(self) -> Optional[str]:
        """Get the least loaded healthy node"""
        healthy_nodes = self.get_healthy_nodes()
        if not healthy_nodes:
            return None
        
        with self.node_lock:
            return min(healthy_nodes, key=lambda n: self.nodes[n].get('active_jobs', 0))

    def assign_job(self, job_id: str, node_id: str) -> bool:
        """Assign job to specific node"""
        with self.node_lock:
            if node_id not in self.nodes:
                return False
            
            if node_id in self.active_jobs.values():
                return False
            
            self.active_jobs[job_id] = node_id
            self.nodes[node_id]['active_jobs'] += 1
            logging.info(f"Assigned job {job_id} to node {node_id}")
            return True

    def complete_job(self, job_id: str) -> bool:
        """Mark job as completed"""
        with self.node_lock:
            if job_id in self.active_jobs:
                node_id = self.active_jobs[job_id]
                del self.active_jobs[job_id]
                
                if node_id in self.nodes:
                    self.nodes[node_id]['active_jobs'] = max(0, self.nodes[node_id]['active_jobs'] - 1)
                
                logging.info(f"Completed job {job_id} on node {node_id}")
                return True
            return False


class LoadBalancer:
    """Intelligent load balancer for processing requests"""

    def __init__(self, processing_manager: DistributedProcessingManager):
        self.processing_manager = processing_manager
        self.request_counts: Dict[str, int] = {}
        self.node_performance: Dict[str, Dict[str, float]] = {}

    def select_node(self, job_requirements: Dict[str, Any]) -> Optional[str]:
        """Select optimal node for job based on requirements and performance"""
        healthy_nodes = self.processing_manager.get_healthy_nodes()
        if not healthy_nodes:
            return None
        
        # Score nodes based on multiple factors
        node_scores = {}
        
        for node_id in healthy_nodes:
            score = 0.0
            
            # Factor 1: Current load (inverted - lower load = higher score)
            node_info = self.processing_manager.nodes.get(node_id, {})
            current_load = node_info.get('active_jobs', 0)
            score += 100 / (current_load + 1)  # Load balancing
            
            # Factor 2: Historical performance
            performance = self.node_performance.get(node_id, {})
            avg_completion_time = performance.get('avg_completion_time', 60.0)
            success_rate = performance.get('success_rate', 0.9)
            score += success_rate * 50  # Performance weighting
            score += 100 / (avg_completion_time + 1)  # Speed weighting
            
            # Factor 3: Request distribution (fairness)
            request_count = self.request_counts.get(node_id, 0)
            score += 50 / (request_count + 1)  # Fair distribution
            
            # Factor 4: Node capabilities vs job requirements
            node_capabilities = node_info.get('capabilities', {})
            if job_requirements.get('gpu_required', False) and not node_capabilities.get('has_gpu', False):
                score -= 100  # Penalize nodes without required capabilities
            
            node_scores[node_id] = score
        
        # Select node with highest score
        if node_scores:
            best_node = max(node_scores, key=node_scores.get)
            self.request_counts[best_node] = self.request_counts.get(best_node, 0) + 1
            return best_node
        
        return None

    def update_node_performance(self, node_id: str, completion_time: float, success: bool):
        """Update node performance metrics"""
        if node_id not in self.node_performance:
            self.node_performance[node_id] = {
                'completion_times': [],
                'success_count': 0,
                'total_jobs': 0
            }
        
        perf = self.node_performance[node_id]
        perf['completion_times'].append(completion_time)
        perf['total_jobs'] += 1
        if success:
            perf['success_count'] += 1
        
        # Keep only recent data (last 100 jobs)
        if len(perf['completion_times']) > 100:
            perf['completion_times'] = perf['completion_times'][-100:]
        
        # Update metrics
        perf['avg_completion_time'] = sum(perf['completion_times']) / len(perf['completion_times'])
        perf['success_rate'] = perf['success_count'] / perf['total_jobs']


# API Models
if FASTAPI_AVAILABLE:
    class JobSubmissionModel(BaseModel):
        input_file: str
        preset: str = "speech_clean"
        quality: str = "high_quality"
        output_format: str = "mp3"
        bitrate: Optional[int] = None
        
    class JobResponseModel(BaseModel):
        job_id: str
        status: JobStatus
        message: str
        result_url: Optional[str] = None

    class AnalysisRequestModel(BaseModel):
        file_path: str

    class HealthResponseModel(BaseModel):
        status: str
        version: str
        nodes: int
        active_jobs: int
        queue_size: int


class PureSoundAPI:
    """Main API server for Pure Sound"""

    def __init__(self):
        self.app = None
        self.security = HTTPBearer()
        self.active_jobs: Dict[str, APIJob] = {}
        self.job_lock = threading.RLock()
        
        # Initialize components
        self.processing_manager = DistributedProcessingManager()
        self.load_balancer = LoadBalancer(self.processing_manager)
        
        # Cloud configuration
        self.cloud_config = CloudConfig(
            provider=CloudProvider.AWS,
            region=os.environ.get('PURE_SOUND_AWS_REGION', 'us-east-1'),
            bucket_name=os.environ.get('PURE_SOUND_S3_BUCKET')
        )
        
        self.cloud_storage = CloudStorageManager(self.cloud_config)
        
        if FASTAPI_AVAILABLE:
            self._create_fastapi_app()
        
        logging.info("Pure Sound API initialized")

    def _create_fastapi_app(self):
        """Create FastAPI application"""
        self.app = FastAPI(
            title="Pure Sound API",
            description="Enterprise Audio Processing API",
            version="1.0.0"
        )
        
        # Security middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure properly in production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", "*.puresound.local"]
        )
        
        # API routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""
        @self.app.get("/", response_model=HealthResponseModel)
        async def health_check():
            """Health check endpoint"""
            return HealthResponseModel(
                status="healthy",
                version="1.0.0",
                nodes=len(self.processing_manager.get_healthy_nodes()),
                active_jobs=len(self.processing_manager.active_jobs),
                queue_size=len(self.processing_manager.job_queue)
            )
        
        @self.app.post("/api/v1/jobs", response_model=JobResponseModel)
        async def submit_job(job: JobSubmissionModel, 
                           credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """Submit audio processing job"""
            # Verify authentication
            if not self._verify_api_key(credentials.credentials):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            # Create job
            job_id = str(uuid.uuid4())
            
            # Validate input file
            if not os.path.exists(job.input_file):
                raise HTTPException(status_code=400, detail="Input file not found")
            
            # Create API job
            api_job = APIJob(
                job_id=job_id,
                input_file=job.input_file,
                output_files=[],  # Will be determined based on processing
                preset=job.preset,
                quality=job.quality,
                status=JobStatus.PENDING
            )
            
            with self.job_lock:
                self.active_jobs[job_id] = api_job
            
            # Assign to processing node
            job_requirements = {'gpu_required': False}
            selected_node = self.load_balancer.select_node(job_requirements)
            
            if selected_node:
                # Start processing asynchronously
                asyncio.create_task(self._process_job_async(job_id, job, selected_node))
                
                return JobResponseModel(
                    job_id=job_id,
                    status=JobStatus.PENDING,
                    message="Job submitted successfully"
                )
            else:
                return JobResponseModel(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    message="No available processing nodes"
                )
        
        @self.app.get("/api/v1/jobs/{job_id}")
        async def get_job_status(job_id: str,
                                credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """Get job status"""
            if not self._verify_api_key(credentials.credentials):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            with self.job_lock:
                if job_id not in self.active_jobs:
                    raise HTTPException(status_code=404, detail="Job not found")
                
                job = self.active_jobs[job_id]
                
                return {
                    "job_id": job.job_id,
                    "status": job.status.value,
                    "progress": job.progress,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "completed_at": job.completed_at,
                    "error_message": job.error_message,
                    "result_url": job.result_url
                }
        
        @self.app.delete("/api/v1/jobs/{job_id}")
        async def cancel_job(job_id: str,
                           credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """Cancel job"""
            if not self._verify_api_key(credentials.credentials):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            with self.job_lock:
                if job_id not in self.active_jobs:
                    raise HTTPException(status_code=404, detail="Job not found")
                
                job = self.active_jobs[job_id]
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    raise HTTPException(status_code=400, detail="Cannot cancel completed job")
                
                job.status = JobStatus.CANCELLED
            
            return {"message": "Job cancelled successfully"}
        
        @self.app.post("/api/v1/analyze")
        async def analyze_audio(request: AnalysisRequestModel,
                              credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """Analyze audio file"""
            if not self._verify_api_key(credentials.credentials):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            try:
                result = audio_analysis_engine.analyze_file(request.file_path)
                if result:
                    return {
                        "content_type": result.content_type.value,
                        "confidence": result.confidence,
                        "quality": result.quality.value,
                        "duration": result.duration,
                        "sample_rate": result.sample_rate,
                        "channels": result.channels,
                        "recommended_format": result.recommended_format,
                        "recommended_bitrates": result.recommended_bitrates,
                        "processing_steps": result.processing_steps
                    }
                else:
                    raise HTTPException(status_code=500, detail="Analysis failed")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/presets")
        async def get_presets(credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """Get available processing presets"""
            if not self._verify_api_key(credentials.credentials):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            try:
                presets = audio_processing_engine.get_available_presets()
                return {"presets": presets}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/nodes")
        async def get_nodes(credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """Get processing nodes status"""
            if not self._verify_api_key(credentials.credentials):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            try:
                nodes = []
                for node_id, node_info in self.processing_manager.nodes.items():
                    nodes.append({
                        "node_id": node_id,
                        "status": node_info.get('status', 'unknown'),
                        "active_jobs": node_info.get('active_jobs', 0),
                        "capabilities": node_info.get('capabilities', {}),
                        "last_heartbeat": node_info.get('last_heartbeat', 0)
                    })
                
                return {"nodes": nodes}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    async def _process_job_async(self, job_id: str, job: 'JobSubmissionModel', node_id: str):
        """Process job asynchronously"""
        try:
            start_time = time.time()
            
            # Update job status
            with self.job_lock:
                self.active_jobs[job_id].status = JobStatus.RUNNING
                self.active_jobs[job_id].started_at = time.time()
            
            # Assign job to node
            self.processing_manager.assign_job(job_id, node_id)
            
            # Create processing job
            output_dir = "/tmp/pure_sound_output"
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f"processed_{os.path.basename(job.input_file)}")
            
            processing_job = audio_processing_engine.create_processing_job(
                job.input_file,
                [output_file],
                job.preset
            )
            
            # Process job
            result = audio_processing_engine.process_job_sync(processing_job)
            
            # Upload results to cloud if configured
            result_url = None
            if result["success"] and self.cloud_storage:
                cloud_path = f"processed/{job_id}/{os.path.basename(output_file)}"
                if self.cloud_storage.upload_file(output_file, cloud_path):
                    # Generate signed URL or return cloud path
                    result_url = f"s3://{self.cloud_config.bucket_name}/{cloud_path}"
            
            # Update job completion
            completion_time = time.time() - start_time
            
            with self.job_lock:
                self.active_jobs[job_id].status = JobStatus.COMPLETED if result["success"] else JobStatus.FAILED
                self.active_jobs[job_id].completed_at = time.time()
                self.active_jobs[job_id].result_url = result_url
                if not result["success"]:
                    self.active_jobs[job_id].error_message = result.get("error", "Unknown error")
            
            # Update node performance
            self.load_balancer.update_node_performance(node_id, completion_time, result["success"])
            
            # Cleanup
            self.processing_manager.complete_job(job_id)
            
            if result["success"]:
                logging.info(f"Job {job_id} completed successfully in {completion_time:.2f}s")
            else:
                logging.error(f"Job {job_id} failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logging.error(f"Job processing error {job_id}: {e}")
            
            with self.job_lock:
                self.active_jobs[job_id].status = JobStatus.FAILED
                self.active_jobs[job_id].completed_at = time.time()
                self.active_jobs[job_id].error_message = str(e)
            
            self.processing_manager.complete_job(job_id)
            self.load_balancer.update_node_performance(node_id, 0, False)

    def _verify_api_key(self, api_key: str) -> bool:
        """Verify API key (simplified implementation)"""
        # In production, this would check against a database of valid API keys
        # For now, use a simple check or environment variable
        valid_key = os.environ.get('PURE_SOUND_API_KEY', 'demo_key')
        return api_key == valid_key

    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Run the API server"""
        if not FASTAPI_AVAILABLE:
            logging.error("FastAPI not available, cannot run API server")
            return
        
        import uvicorn
        logging.info(f"Starting Pure Sound API server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, debug=debug)


# Docker orchestration support
class DockerOrchestrator:
    """Docker-based orchestration for distributed processing"""

    def __init__(self):
        self.client = None
        if DOCKER_AVAILABLE:
            try:
                self.client = docker.from_env()
            except Exception as e:
                logging.error(f"Failed to initialize Docker client: {e}")

    def scale_processing_nodes(self, target_count: int) -> bool:
        """Scale processing nodes up or down"""
        if not self.client:
            return False
        
        try:
            # Get current containers
            containers = self.client.containers.list(filters={"label": "puresound.processing-node"})
            current_count = len(containers)
            
            if current_count < target_count:
                # Scale up
                for i in range(target_count - current_count):
                    container = self.client.containers.run(
                        "puresound-processing-node:latest",
                        detach=True,
                        labels={
                            "puresound.processing-node": "true",
                            "puresound.node-id": str(uuid.uuid4())
                        },
                        environment={
                            "NODE_TYPE": "processing",
                            "API_SERVER": os.environ.get('API_SERVER_HOST', 'localhost:8000')
                        }
                    )
                    logging.info(f"Scaled up: started processing node {container.short_id}")
            
            elif current_count > target_count:
                # Scale down
                containers_to_stop = containers[target_count:]
                for container in containers_to_stop:
                    container.stop()
                    container.remove()
                    logging.info(f"Scaled down: stopped processing node {container.short_id}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to scale processing nodes: {e}")
            return False

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status"""
        if not self.client:
            return {"error": "Docker not available"}
        
        try:
            processing_containers = self.client.containers.list(
                filters={"label": "puresound.processing-node"}
            )
            
            return {
                "processing_nodes": len(processing_containers),
                "containers": [
                    {
                        "id": c.short_id,
                        "status": c.status,
                        "image": c.image.tags[0] if c.image.tags else c.image.id,
                        "created": c.attrs['Created']
                    }
                    for c in processing_containers
                ]
            }
            
        except Exception as e:
            return {"error": str(e)}


# Global API instance
api_server = PureSoundAPI()
docker_orchestrator = DockerOrchestrator()

def run_api_server():
    """Run the API server"""
    api_server.run(
        host=os.environ.get('PURE_SOUND_API_HOST', '0.0.0.0'),
        port=int(os.environ.get('PURE_SOUND_API_PORT', '8000')),
        debug=os.environ.get('PURE_SOUND_DEBUG', 'false').lower() == 'true'
    )

if __name__ == "__main__":
    run_api_server()