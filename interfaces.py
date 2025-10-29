"""
Standardized Interfaces for Module Interaction

This module defines abstract base classes and protocols that standardize
how modules interact with each other, promoting consistency and testability.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Protocol, Union
from pathlib import Path
import threading


class IServiceProvider(Protocol):
    """Protocol for service providers"""

    def get_service(self, service_type: type) -> Any:
        """Get a service instance by type"""
        ...

    def has_service(self, service_type: type) -> bool:
        """Check if a service is available"""
        ...


class IEventPublisher(Protocol):
    """Protocol for event publishing"""

    def publish_event(self, name: str, data: Dict[str, Any], source: str) -> None:
        """Publish an event"""
        ...


class IConfigurable(Protocol):
    """Protocol for configurable components"""

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the component"""
        ...

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration"""
        ...


class ILifecycle(Protocol):
    """Protocol for components with lifecycle management"""

    def initialize(self) -> None:
        """Initialize the component"""
        ...

    def shutdown(self) -> None:
        """Shutdown the component"""
        ...


# Abstract Base Classes

class AudioProcessor(ABC):
    """Abstract base class for audio processing components"""

    def __init__(self, service_provider: IServiceProvider):
        self._service_provider = service_provider
        self._event_publisher = service_provider.get_service(IEventPublisher)
        self._config = {}
        self._initialized = False

    @abstractmethod
    def process_file(self, input_path: str, output_path: str, **kwargs) -> bool:
        """Process a single audio file"""
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats"""
        pass

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the processor"""
        self._config.update(config)
        self._publish_event("processor.configured", {
            "processor": self.__class__.__name__,
            "config": config
        })

    def initialize(self) -> None:
        """Initialize the processor"""
        if not self._initialized:
            self._initialized = True
            self._publish_event("processor.initialized", {
                "processor": self.__class__.__name__
            })

    def shutdown(self) -> None:
        """Shutdown the processor"""
        if self._initialized:
            self._initialized = False
            self._publish_event("processor.shutdown", {
                "processor": self.__class__.__name__
            })

    def _publish_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Publish an event"""
        if self._event_publisher:
            self._event_publisher.publish_event(
                event_name,
                data,
                self.__class__.__name__
            )


class StorageProvider(ABC):
    """Abstract base class for storage providers"""

    def __init__(self, service_provider: IServiceProvider):
        self._service_provider = service_provider
        self._event_publisher = service_provider.get_service(IEventPublisher)
        self._config = {}
        self._initialized = False

    @abstractmethod
    def store_file(self, local_path: str, key: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a file"""
        pass

    @abstractmethod
    def retrieve_file(self, key: str, local_path: str) -> bool:
        """Retrieve a file"""
        pass

    @abstractmethod
    def delete_file(self, key: str) -> bool:
        """Delete a file"""
        pass

    @abstractmethod
    def list_files(self, prefix: str = "") -> List[str]:
        """List files with optional prefix"""
        pass

    @abstractmethod
    def get_file_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get file information"""
        pass

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the storage provider"""
        self._config.update(config)
        self._publish_event("storage.configured", {
            "provider": self.__class__.__name__,
            "config": config
        })

    def initialize(self) -> None:
        """Initialize the storage provider"""
        if not self._initialized:
            self._initialized = True
            self._publish_event("storage.initialized", {
                "provider": self.__class__.__name__
            })

    def shutdown(self) -> None:
        """Shutdown the storage provider"""
        if self._initialized:
            self._initialized = False
            self._publish_event("storage.shutdown", {
                "provider": self.__class__.__name__
            })

    def _publish_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Publish an event"""
        if self._event_publisher:
            self._event_publisher.publish_event(
                event_name,
                data,
                self.__class__.__name__
            )


class JobProcessor(ABC):
    """Abstract base class for job processing components"""

    def __init__(self, service_provider: IServiceProvider):
        self._service_provider = service_provider
        self._event_publisher = service_provider.get_service(IEventPublisher)
        self._config = {}
        self._initialized = False
        self._running_jobs: Dict[str, threading.Thread] = {}

    @abstractmethod
    def submit_job(self, job_data: Dict[str, Any]) -> str:
        """Submit a job for processing"""
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        pass

    @abstractmethod
    def get_active_jobs(self) -> List[str]:
        """Get list of active job IDs"""
        pass

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the job processor"""
        self._config.update(config)
        self._publish_event("job_processor.configured", {
            "processor": self.__class__.__name__,
            "config": config
        })

    def initialize(self) -> None:
        """Initialize the job processor"""
        if not self._initialized:
            self._initialized = True
            self._publish_event("job_processor.initialized", {
                "processor": self.__class__.__name__
            })

    def shutdown(self) -> None:
        """Shutdown the job processor"""
        if self._initialized:
            # Cancel all running jobs
            for job_id in list(self._running_jobs.keys()):
                self.cancel_job(job_id)

            self._initialized = False
            self._publish_event("job_processor.shutdown", {
                "processor": self.__class__.__name__
            })

    def _publish_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Publish an event"""
        if self._event_publisher:
            self._event_publisher.publish_event(
                event_name,
                data,
                self.__class__.__name__
            )


class ConfigurationProvider(ABC):
    """Abstract base class for configuration providers"""

    def __init__(self, service_provider: IServiceProvider):
        self._service_provider = service_provider
        self._event_publisher = service_provider.get_service(IEventPublisher)
        self._config = {}
        self._initialized = False

    @abstractmethod
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a configuration setting"""
        pass

    @abstractmethod
    def set_setting(self, key: str, value: Any) -> None:
        """Set a configuration setting"""
        pass

    @abstractmethod
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a configuration section"""
        pass

    @abstractmethod
    def save_config(self) -> None:
        """Save configuration to persistent storage"""
        pass

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the configuration provider"""
        self._config.update(config)
        self._publish_event("config.configured", {
            "provider": self.__class__.__name__,
            "config": config
        })

    def initialize(self) -> None:
        """Initialize the configuration provider"""
        if not self._initialized:
            self._initialized = True
            self._publish_event("config.initialized", {
                "provider": self.__class__.__name__
            })

    def shutdown(self) -> None:
        """Shutdown the configuration provider"""
        if self._initialized:
            self.save_config()
            self._initialized = False
            self._publish_event("config.shutdown", {
                "provider": self.__class__.__name__
            })

    def _publish_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Publish an event"""
        if self._event_publisher:
            self._event_publisher.publish_event(
                event_name,
                data,
                self.__class__.__name__
            )


class AnalysisProvider(ABC):
    """Abstract base class for audio analysis components"""

    def __init__(self, service_provider: IServiceProvider):
        self._service_provider = service_provider
        self._event_publisher = service_provider.get_service(IEventPublisher)
        self._config = {}
        self._initialized = False

    @abstractmethod
    def analyze_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Analyze an audio file"""
        pass

    @abstractmethod
    def get_quick_stats(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get quick statistics for a file"""
        pass

    @abstractmethod
    def get_supported_analysis_types(self) -> List[str]:
        """Get list of supported analysis types"""
        pass

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the analysis provider"""
        self._config.update(config)
        self._publish_event("analysis.configured", {
            "provider": self.__class__.__name__,
            "config": config
        })

    def initialize(self) -> None:
        """Initialize the analysis provider"""
        if not self._initialized:
            self._initialized = True
            self._publish_event("analysis.initialized", {
                "provider": self.__class__.__name__
            })

    def shutdown(self) -> None:
        """Shutdown the analysis provider"""
        if self._initialized:
            self._initialized = False
            self._publish_event("analysis.shutdown", {
                "provider": self.__class__.__name__
            })

    def _publish_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Publish an event"""
        if self._event_publisher:
            self._event_publisher.publish_event(
                event_name,
                data,
                self.__class__.__name__
            )


# Data Transfer Objects

class AudioFileInfo:
    """Data transfer object for audio file information"""

    def __init__(self, path: Union[str, Path], **kwargs):
        self.path = Path(path)
        self.codec: Optional[str] = kwargs.get('codec')
        self.sample_rate: Optional[int] = kwargs.get('sample_rate')
        self.channels: Optional[int] = kwargs.get('channels')
        self.duration: Optional[float] = kwargs.get('duration')
        self.bitrate: Optional[int] = kwargs.get('bitrate')
        self.size_bytes: Optional[int] = kwargs.get('size_bytes')
        self.format_name: Optional[str] = kwargs.get('format_name')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'path': str(self.path),
            'codec': self.codec,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'duration': self.duration,
            'bitrate': self.bitrate,
            'size_bytes': self.size_bytes,
            'format_name': self.format_name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioFileInfo':
        """Create from dictionary"""
        return cls(**data)


class CompressionJob:
    """Data transfer object for compression jobs"""

    def __init__(self, job_id: str, input_file: str, output_file: str, **kwargs):
        self.job_id = job_id
        self.input_file = input_file
        self.output_file = output_file
        self.bitrate: int = kwargs.get('bitrate', 128)
        self.format: str = kwargs.get('format', 'mp3')
        self.filter_chain: Optional[str] = kwargs.get('filter_chain')
        self.channels: int = kwargs.get('channels', 1)
        self.preserve_metadata: bool = kwargs.get('preserve_metadata', True)
        self.priority: int = kwargs.get('priority', 2)
        self.status: str = kwargs.get('status', 'pending')
        self.progress: float = kwargs.get('progress', 0.0)
        self.error_message: Optional[str] = kwargs.get('error_message')
        self.created_at: float = kwargs.get('created_at', 0.0)
        self.started_at: Optional[float] = kwargs.get('started_at')
        self.completed_at: Optional[float] = kwargs.get('completed_at')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'job_id': self.job_id,
            'input_file': self.input_file,
            'output_file': self.output_file,
            'bitrate': self.bitrate,
            'format': self.format,
            'filter_chain': self.filter_chain,
            'channels': self.channels,
            'preserve_metadata': self.preserve_metadata,
            'priority': self.priority,
            'status': self.status,
            'progress': self.progress,
            'error_message': self.error_message,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompressionJob':
        """Create from dictionary"""
        return cls(**data)


class AnalysisResult:
    """Data transfer object for analysis results"""

    def __init__(self, file_path: str, **kwargs):
        self.file_path = file_path
        self.content_type: Optional[str] = kwargs.get('content_type')
        self.dynamic_range: Optional[float] = kwargs.get('dynamic_range')
        self.speech_probability: Optional[float] = kwargs.get('speech_probability')
        self.music_probability: Optional[float] = kwargs.get('music_probability')
        self.recommended_format: Optional[str] = kwargs.get('recommended_format')
        self.recommended_bitrates: List[int] = kwargs.get('recommended_bitrates', [])
        self.enable_compression: bool = kwargs.get('enable_compression', False)
        self.enable_loudnorm: bool = kwargs.get('enable_loudnorm', True)
        self.reasoning: List[str] = kwargs.get('reasoning', [])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'file_path': self.file_path,
            'content_type': self.content_type,
            'dynamic_range': self.dynamic_range,
            'speech_probability': self.speech_probability,
            'music_probability': self.music_probability,
            'recommended_format': self.recommended_format,
            'recommended_bitrates': self.recommended_bitrates,
            'enable_compression': self.enable_compression,
            'enable_loudnorm': self.enable_loudnorm,
            'reasoning': self.reasoning
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create from dictionary"""
        return cls(**data)