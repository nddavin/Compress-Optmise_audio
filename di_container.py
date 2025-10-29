"""
Dependency Injection Container and Service Locator

This module provides a centralized dependency injection container and service locator
pattern implementation for managing component dependencies and lifecycles.
"""

import threading
import logging
from typing import Dict, Any, Type, TypeVar, Optional, Callable, List
from weakref import WeakValueDictionary
import inspect

T = TypeVar('T')


class ServiceLifetime:
    """Service lifetime options"""
    SINGLETON = "singleton"  # One instance for entire application
    TRANSIENT = "transient"  # New instance each time requested
    SCOPED = "scoped"       # One instance per scope (not implemented yet)


class ServiceDescriptor:
    """Descriptor for a registered service"""

    def __init__(self, service_type: Type, implementation_type: Optional[Type] = None,
                 factory: Optional[Callable] = None, lifetime: str = ServiceLifetime.SINGLETON,
                 instance: Optional[Any] = None):
        self.service_type = service_type
        self.implementation_type = implementation_type or service_type
        self.factory = factory
        self.lifetime = lifetime
        self.instance = instance
        self._lock = threading.Lock()

    def get_instance(self, container: 'DIContainer') -> Any:
        """Get or create service instance based on lifetime"""
        if self.lifetime == ServiceLifetime.SINGLETON:
            if self.instance is None:
                with self._lock:
                    if self.instance is None:  # Double-check locking
                        self.instance = self._create_instance(container)
            return self.instance

        elif self.lifetime == ServiceLifetime.TRANSIENT:
            return self._create_instance(container)

        else:
            raise ValueError(f"Unsupported service lifetime: {self.lifetime}")

    def _create_instance(self, container: 'DIContainer') -> Any:
        """Create a new instance of the service"""
        if self.factory:
            return self.factory(container)

        # Auto-inject dependencies based on constructor parameters
        constructor = self.implementation_type.__init__
        sig = inspect.signature(constructor)

        # Skip 'self' parameter
        params = list(sig.parameters.values())[1:]

        kwargs = {}
        for param in params:
            if param.name in ['service_provider', 'container']:
                # Special case: inject the container itself
                kwargs[param.name] = container
            elif param.annotation != inspect.Parameter.empty:
                # Try to resolve dependency by type
                try:
                    kwargs[param.name] = container.get_service(param.annotation)
                except KeyError:
                    # If dependency not found, check if it has a default value
                    if param.default != inspect.Parameter.empty:
                        kwargs[param.name] = param.default
                    else:
                        raise ValueError(f"Cannot resolve dependency '{param.name}' of type {param.annotation} for {self.implementation_type}")
            else:
                # Skip untyped parameters that have defaults (like *args, **kwargs)
                if param.default != inspect.Parameter.empty:
                    continue
                # For other untyped parameters, try to skip them if they're common Python parameters
                if param.name in ['args', 'kwargs']:
                    continue
                raise ValueError(f"Cannot resolve untyped parameter '{param.name}' for {self.implementation_type}")

        return self.implementation_type(**kwargs)


class DIContainer:
    """
    Dependency Injection Container

    Manages service registration and resolution with support for different lifetimes.
    """

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._lock = threading.RLock()
        self._event_publisher = None

    def register_singleton(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                          instance: Optional[T] = None) -> None:
        """Register a singleton service"""
        self._register_service(service_type, implementation_type, ServiceLifetime.SINGLETON, instance)

    def register_transient(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> None:
        """Register a transient service"""
        self._register_service(service_type, implementation_type, ServiceLifetime.TRANSIENT)

    def register_factory(self, service_type: Type[T], factory: Callable[['DIContainer'], T],
                        lifetime: str = ServiceLifetime.SINGLETON) -> None:
        """Register a service with a custom factory function"""
        descriptor = ServiceDescriptor(service_type, factory=factory, lifetime=lifetime)
        self._services[service_type] = descriptor
        self._publish_registration_event(service_type, lifetime)

    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """Register a pre-created instance as a singleton"""
        descriptor = ServiceDescriptor(service_type, instance=instance, lifetime=ServiceLifetime.SINGLETON)
        self._services[service_type] = descriptor
        self._publish_registration_event(service_type, ServiceLifetime.SINGLETON)

    def _register_service(self, service_type: Type[T], implementation_type: Optional[Type[T]],
                         lifetime: str, instance: Optional[T] = None) -> None:
        """Internal service registration"""
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type,
                lifetime=lifetime,
                instance=instance
            )
            self._services[service_type] = descriptor
            self._publish_registration_event(service_type, lifetime)

    def get_service(self, service_type: Type[T]) -> T:
        """Get a service instance"""
        with self._lock:
            if service_type not in self._services:
                raise KeyError(f"Service {service_type} not registered")

            descriptor = self._services[service_type]
            instance = descriptor.get_instance(self)

            # Publish service resolution event
            self._publish_event("service.resolved", {
                "service_type": service_type.__name__,
                "lifetime": descriptor.lifetime
            })

            return instance

    def has_service(self, service_type: Type[T]) -> bool:
        """Check if a service is registered"""
        with self._lock:
            return service_type in self._services

    def get_registered_services(self) -> List[Type]:
        """Get list of registered service types"""
        with self._lock:
            return list(self._services.keys())

    def unregister_service(self, service_type: Type[T]) -> bool:
        """Unregister a service"""
        with self._lock:
            if service_type in self._services:
                del self._services[service_type]
                self._publish_event("service.unregistered", {
                    "service_type": service_type.__name__
                })
                return True
        return False

    def clear(self) -> None:
        """Clear all registered services"""
        with self._lock:
            self._services.clear()
            self._publish_event("container.cleared", {})

    def set_event_publisher(self, publisher) -> None:
        """Set the event publisher for service lifecycle events"""
        self._event_publisher = publisher

    def _publish_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Publish an event if publisher is available"""
        if self._event_publisher:
            self._event_publisher.publish_event(event_name, data, "DIContainer")

    def _publish_registration_event(self, service_type: Type, lifetime: str) -> None:
        """Publish service registration event"""
        self._publish_event("service.registered", {
            "service_type": service_type.__name__,
            "lifetime": lifetime
        })


class ServiceLocator:
    """
    Service Locator pattern implementation

    Provides a global registry for services that can be accessed statically.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._services: WeakValueDictionary[Type, Any] = WeakValueDictionary()
        self._factories: Dict[Type, Callable[[], Any]] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'ServiceLocator':
        """Get the singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_service(self, service_type: Type[T], instance: T) -> None:
        """Register a service instance"""
        with self._lock:
            self._services[service_type] = instance
            logging.debug(f"Registered service: {service_type.__name__}")

    def register_factory(self, service_type: Type[T], factory: Callable[[], T]) -> None:
        """Register a service factory"""
        with self._lock:
            self._factories[service_type] = factory
            logging.debug(f"Registered factory for: {service_type.__name__}")

    def get_service(self, service_type: Type[T]) -> T:
        """Get a service instance"""
        with self._lock:
            # Try to get existing instance
            if service_type in self._services:
                return self._services[service_type]

            # Try to create from factory
            if service_type in self._factories:
                instance = self._factories[service_type]()
                self._services[service_type] = instance
                return instance

            raise KeyError(f"Service {service_type} not registered")

    def has_service(self, service_type: Type[T]) -> bool:
        """Check if a service is available"""
        with self._lock:
            return service_type in self._services or service_type in self._factories

    def unregister_service(self, service_type: Type[T]) -> bool:
        """Unregister a service"""
        with self._lock:
            if service_type in self._services:
                del self._services[service_type]
                return True
            if service_type in self._factories:
                del self._factories[service_type]
                return True
        return False

    def get_registered_services(self) -> List[Type]:
        """Get list of registered service types"""
        with self._lock:
            return list(set(list(self._services.keys()) + list(self._factories.keys())))


# Global instances with optimized initialization
di_container = DIContainer()
service_locator = ServiceLocator.get_instance()

# Register resource pools for optimized access
def initialize_resource_pools():
    """Initialize resource pools for better performance"""
    from resource_pool import resource_pool_manager

    # Register resource pools for common services
    resource_pool_manager.register_loader("di_container", lambda: di_container)
    resource_pool_manager.register_loader("service_locator", lambda: service_locator)

try:
    initialize_resource_pools()
except ImportError:
    # Resource pool not available yet
    pass


# Convenience functions for global access
def register_singleton(service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                      instance: Optional[T] = None) -> None:
    """Register a singleton service globally"""
    di_container.register_singleton(service_type, implementation_type, instance)


def register_transient(service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> None:
    """Register a transient service globally"""
    di_container.register_transient(service_type, implementation_type)


def get_service(service_type: Type[T]) -> T:
    """Get a service instance globally"""
    return di_container.get_service(service_type)


def has_service(service_type: Type[T]) -> bool:
    """Check if a service is available globally"""
    return di_container.has_service(service_type)


# Initialize the container with core services
def initialize_core_services():
    """Initialize core services in the DI container with lazy loading"""

    # Lazy load and register the event bus
    def load_event_bus(container):
        from events import event_bus
        return event_bus

    di_container.register_factory(type(load_event_bus(None)), load_event_bus)

    # Lazy load and register service locator
    def load_service_locator(container):
        return service_locator

    di_container.register_factory(ServiceLocator, load_service_locator)

    # Register the DI container itself for self-injection
    di_container.register_instance(DIContainer, di_container)

    # Lazy load and register core interfaces
    def load_interfaces(container):
        from interfaces import IEventPublisher, IServiceProvider
        return {
            'event_publisher': load_event_bus(container),
            'service_provider': di_container
        }

    # Register interfaces with lazy loading
    interfaces_data = load_interfaces(di_container)
    di_container.register_instance(interfaces_data['event_publisher'].__class__, interfaces_data['event_publisher'])
    di_container.register_instance(interfaces_data['service_provider'].__class__, interfaces_data['service_provider'])

    logging.info("Core services initialized with lazy loading")


# Auto-initialize on import with lazy loading
try:
    initialize_core_services()
except ImportError:
    # Dependencies not available yet, will be initialized later
    pass