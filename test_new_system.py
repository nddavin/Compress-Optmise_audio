"""
Unit tests for the new cross-module communication system

Tests the event system, dependency injection, and standardized interfaces.
"""

import unittest
import time
from unittest.mock import Mock, MagicMock
import threading
from typing import Dict, Any, List

# Import new system components
from events import EventBus, Event, EventPriority, event_bus, publish_event
from interfaces import (
    IServiceProvider, IEventPublisher, AudioProcessor, StorageProvider,
    AudioFileInfo, CompressionJob, AnalysisResult
)
from di_container import DIContainer, ServiceLocator, di_container, service_locator


class TestEventSystem(unittest.TestCase):
    """Test the event bus and pub-sub system"""

    def setUp(self):
        self.event_bus = EventBus(max_history=50)
        self.received_events = []
        self.event_lock = threading.Lock()

    def event_callback(self, event: Event):
        with self.event_lock:
            self.received_events.append(event)

    def test_event_creation(self):
        """Test event creation and properties"""
        event = Event(
            name="test.event",
            data={"key": "value"},
            source="test_source",
            priority=EventPriority.HIGH
        )

        self.assertEqual(event.name, "test.event")
        self.assertEqual(event.data["key"], "value")
        self.assertEqual(event.source, "test_source")
        self.assertEqual(event.priority, EventPriority.HIGH)
        self.assertIsInstance(event.timestamp, float)

    def test_event_subscription_and_publishing(self):
        """Test subscribing to events and publishing them"""
        # Subscribe to an event
        subscription_id = self.event_bus.subscribe("test.event", self.event_callback)

        # Publish an event
        self.event_bus.publish_event("test.event", {"message": "hello"}, "test_publisher")

        # Wait a bit for async processing
        time.sleep(0.1)

        # Check that event was received
        with self.event_lock:
            self.assertEqual(len(self.received_events), 1)
            event = self.received_events[0]
            self.assertEqual(event.name, "test.event")
            self.assertEqual(event.data["message"], "hello")
            self.assertEqual(event.source, "test_publisher")

    def test_event_filtering(self):
        """Test event filtering with custom filter functions"""
        received_filtered = []

        def filter_callback(event: Event):
            received_filtered.append(event)

        # Subscribe with filter that only accepts high priority events
        self.event_bus.subscribe(
            "test.event",
            filter_callback,
            filter_func=lambda e: e.priority == EventPriority.HIGH
        )

        # Publish events with different priorities
        self.event_bus.publish_event("test.event", {"msg": "low"}, "test", EventPriority.LOW)
        self.event_bus.publish_event("test.event", {"msg": "high"}, "test", EventPriority.HIGH)

        time.sleep(0.1)

        # Should only receive the high priority event
        self.assertEqual(len(received_filtered), 1)
        self.assertEqual(received_filtered[0].data["msg"], "high")

    def test_wildcard_subscriptions(self):
        """Test wildcard event subscriptions"""
        wildcard_events = []

        def wildcard_callback(event: Event):
            wildcard_events.append(event)

        # Subscribe to all events starting with "file."
        self.event_bus.subscribe("file.*", wildcard_callback)

        # Publish various events
        self.event_bus.publish_event("file.created", {"path": "/test"}, "test")
        self.event_bus.publish_event("file.deleted", {"path": "/test"}, "test")
        self.event_bus.publish_event("job.started", {"id": "123"}, "test")  # Should not match

        time.sleep(0.1)

        # Should receive 2 file events
        self.assertEqual(len(wildcard_events), 2)
        event_names = [e.name for e in wildcard_events]
        self.assertIn("file.created", event_names)
        self.assertIn("file.deleted", event_names)

    def test_event_history(self):
        """Test event history functionality"""
        # Publish some events
        self.event_bus.publish_event("event1", {"data": 1}, "test")
        self.event_bus.publish_event("event2", {"data": 2}, "test")
        self.event_bus.publish_event("event1", {"data": 3}, "test")

        # Get history
        history = self.event_bus.get_event_history("event1")

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].data["data"], 1)
        self.assertEqual(history[1].data["data"], 3)


class TestDIContainer(unittest.TestCase):
    """Test the dependency injection container"""

    def setUp(self):
        self.container = DIContainer()

    def test_singleton_registration_and_resolution(self):
        """Test singleton service registration and resolution"""

        class TestService:
            def __init__(self):
                self.value = 42

        # Register singleton
        self.container.register_singleton(TestService)

        # Resolve multiple times - should get same instance
        instance1 = self.container.get_service(TestService)
        instance2 = self.container.get_service(TestService)

        self.assertIs(instance1, instance2)
        self.assertEqual(instance1.value, 42)

    def test_transient_registration_and_resolution(self):
        """Test transient service registration and resolution"""

        class TestService:
            def __init__(self):
                self.id = id(self)

        # Register transient
        self.container.register_transient(TestService)

        # Resolve multiple times - should get different instances
        instance1 = self.container.get_service(TestService)
        instance2 = self.container.get_service(TestService)

        self.assertIsNot(instance1, instance2)
        self.assertNotEqual(instance1.id, instance2.id)

    def test_factory_registration(self):
        """Test factory-based service registration"""

        class TestService:
            def __init__(self, value: int):
                self.value = value

        def service_factory(container):
            return TestService(100)

        # Register factory
        self.container.register_factory(TestService, service_factory)

        # Resolve
        instance = self.container.get_service(TestService)

        self.assertEqual(instance.value, 100)

    def test_dependency_injection(self):
        """Test automatic dependency injection"""

        class Logger:
            def log(self, message: str):
                return f"LOG: {message}"

        class Database:
            def __init__(self, logger: Logger):
                self.logger = logger

            def save(self, data: str):
                return self.logger.log(f"Saving: {data}")

        # Register services
        self.container.register_singleton(Logger)
        self.container.register_singleton(Database)

        # Resolve database - should automatically inject logger
        db = self.container.get_service(Database)

        self.assertIsInstance(db.logger, Logger)
        self.assertEqual(db.save("test"), "LOG: Saving: test")

    def test_service_not_found(self):
        """Test error when service is not registered"""

        class UnknownService:
            pass

        with self.assertRaises(KeyError):
            self.container.get_service(UnknownService)


class TestServiceLocator(unittest.TestCase):
    """Test the service locator pattern"""

    def setUp(self):
        self.locator = ServiceLocator()

    def test_service_registration_and_resolution(self):
        """Test service registration and resolution"""

        class TestService:
            def get_value(self):
                return "test_value"

        # Register service
        service_instance = TestService()
        self.locator.register_service(TestService, service_instance)

        # Resolve service
        resolved = self.locator.get_service(TestService)

        self.assertIs(resolved, service_instance)
        self.assertEqual(resolved.get_value(), "test_value")

    def test_factory_registration(self):
        """Test factory-based service registration"""

        class TestService:
            def __init__(self, value: int):
                self.value = value

        def service_factory():
            return TestService(42)

        # Register factory
        self.locator.register_factory(TestService, service_factory)

        # Resolve service
        resolved = self.locator.get_service(TestService)

        self.assertIsInstance(resolved, TestService)
        self.assertEqual(resolved.value, 42)


class TestInterfaces(unittest.TestCase):
    """Test the standardized interfaces"""

    def setUp(self):
        self.container = DIContainer()
        # Register mock event publisher
        mock_publisher = Mock()
        self.container.register_instance(IEventPublisher, mock_publisher)
        self.container.register_instance(IServiceProvider, self.container)

    def test_audio_processor_interface(self):
        """Test the AudioProcessor abstract base class"""

        class MockAudioProcessor(AudioProcessor):
            def process_file(self, input_path: str, output_path: str, **kwargs) -> bool:
                return True

            def get_supported_formats(self) -> List[str]:
                return ["mp3", "aac"]

        # Create processor
        processor = MockAudioProcessor(self.container)

        # Test interface methods
        self.assertTrue(processor.process_file("input.wav", "output.mp3"))
        self.assertEqual(processor.get_supported_formats(), ["mp3", "aac"])

        # Test configuration
        processor.configure({"format": "aac"})
        self.assertEqual(processor._config["format"], "aac")

    def test_data_transfer_objects(self):
        """Test the data transfer objects"""

        # Test AudioFileInfo
        file_info = AudioFileInfo("/path/to/file.wav", codec="pcm_s16le", sample_rate=44100)
        self.assertEqual(file_info.path.name, "file.wav")
        self.assertEqual(file_info.codec, "pcm_s16le")
        self.assertEqual(file_info.sample_rate, 44100)

        # Convert to/from dict
        data = file_info.to_dict()
        reconstructed = AudioFileInfo.from_dict(data)
        self.assertEqual(reconstructed.codec, "pcm_s16le")

        # Test CompressionJob
        job = CompressionJob("job123", "input.wav", "output.mp3", bitrate=128)
        self.assertEqual(job.job_id, "job123")
        self.assertEqual(job.bitrate, 128)

        # Test AnalysisResult
        result = AnalysisResult("file.wav", content_type="speech", speech_probability=0.9)
        self.assertEqual(result.file_path, "file.wav")
        self.assertEqual(result.content_type, "speech")
        self.assertEqual(result.speech_probability, 0.9)


class TestIntegration(unittest.TestCase):
    """Test integration between components"""

    def test_event_driven_di_container(self):
        """Test that DI container publishes events"""

        # Create container with event bus
        container = DIContainer()
        event_bus = EventBus()
        container.set_event_publisher(event_bus)

        events_received = []

        def event_callback(event: Event):
            events_received.append(event)

        event_bus.subscribe("service.*", event_callback)

        # Register a service
        class TestService:
            pass

        container.register_singleton(TestService)

        # Should have received service registration event
        time.sleep(0.1)
        self.assertTrue(any(e.name == "service.registered" for e in events_received))

        # Resolve service
        container.get_service(TestService)

        # Should have received service resolution event
        time.sleep(0.1)
        self.assertTrue(any(e.name == "service.resolved" for e in events_received))


if __name__ == '__main__':
    unittest.main()