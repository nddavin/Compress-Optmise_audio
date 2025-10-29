"""
Resource Pooling and Connection Caching System

This module provides resource pooling for reusable objects, connection caching for cloud services,
and intelligent initialization strategies to optimize startup time and resource usage.
"""

import threading
import time
import logging
from typing import Dict, Any, Callable, TypeVar, Generic, List, Optional
from weakref import WeakSet, ref
from contextlib import contextmanager
import gc
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logging.warning("psutil not available. Memory monitoring will be limited.")
import os

T = TypeVar('T')

class ResourcePool(Generic[T]):
    """Generic resource pool for managing reusable objects"""

    def __init__(self, factory: Callable[[], T], max_size: int = 10, min_size: int = 1,
                 max_idle_time: float = 300.0, cleanup_interval: float = 60.0):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        self.cleanup_interval = cleanup_interval

        self._pool: List[PoolItem[T]] = []
        self._lock = threading.RLock()
        self._shutdown = False

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()

        # Initialize minimum pool size
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize minimum pool size"""
        for _ in range(self.min_size):
            try:
                resource = self.factory()
                item = PoolItem(resource, time.time())
                self._pool.append(item)
            except Exception as e:
                logging.warning(f"Failed to create initial pool resource: {e}")

    def acquire(self, timeout: float = 30.0) -> T:
        """Acquire a resource from the pool"""
        if self._shutdown:
            raise RuntimeError("Pool is shutting down")

        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._lock:
                # Try to get an available resource
                for item in self._pool:
                    if not item.in_use:
                        item.in_use = True
                        item.last_used = time.time()
                        return item.resource

                # Create new resource if under max size
                if len(self._pool) < self.max_size:
                    try:
                        resource = self.factory()
                        item = PoolItem(resource, time.time(), in_use=True)
                        self._pool.append(item)
                        return resource
                    except Exception as e:
                        logging.warning(f"Failed to create new pool resource: {e}")

            # Wait a bit before retrying
            time.sleep(0.1)

        raise TimeoutError(f"Timeout waiting for resource from pool (timeout={timeout}s)")

    def release(self, resource: T):
        """Release a resource back to the pool"""
        with self._lock:
            for item in self._pool:
                if item.resource is resource:
                    item.in_use = False
                    item.last_used = time.time()
                    break

    @contextmanager
    def get_resource(self, timeout: float = 30.0):
        """Context manager for resource acquisition"""
        resource = self.acquire(timeout)
        try:
            yield resource
        finally:
            self.release(resource)

    def _cleanup_worker(self):
        """Background cleanup worker"""
        while not self._shutdown:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_idle_resources()
            except Exception as e:
                logging.error(f"Error in cleanup worker: {e}")

    def _cleanup_idle_resources(self):
        """Remove idle resources beyond minimum size"""
        with self._lock:
            current_time = time.time()
            idle_items = []

            for item in self._pool:
                if not item.in_use and (current_time - item.last_used) > self.max_idle_time:
                    idle_items.append(item)

            # Keep at least min_size resources
            items_to_remove = max(0, len(idle_items) - self.min_size)
            for i in range(items_to_remove):
                try:
                    # Try to close/cleanup resource if it has a close method
                    if hasattr(idle_items[i].resource, 'close'):
                        idle_items[i].resource.close()
                    elif hasattr(idle_items[i].resource, 'cleanup'):
                        idle_items[i].resource.cleanup()
                except Exception as e:
                    logging.warning(f"Error cleaning up resource: {e}")

                self._pool.remove(idle_items[i])

    def shutdown(self):
        """Shutdown the pool and cleanup all resources"""
        self._shutdown = True
        with self._lock:
            for item in self._pool:
                try:
                    if hasattr(item.resource, 'close'):
                        item.resource.close()
                    elif hasattr(item.resource, 'cleanup'):
                        item.resource.cleanup()
                except Exception as e:
                    logging.warning(f"Error shutting down resource: {e}")
            self._pool.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            total = len(self._pool)
            in_use = sum(1 for item in self._pool if item.in_use)
            available = total - in_use

            return {
                "total_resources": total,
                "available_resources": available,
                "in_use_resources": in_use,
                "max_size": self.max_size,
                "min_size": self.min_size
            }


class PoolItem(Generic[T]):
    """Wrapper for pooled resources"""

    def __init__(self, resource: T, created_at: float, in_use: bool = False):
        self.resource = resource
        self.created_at = created_at
        self.last_used = created_at
        self.in_use = in_use


class ConnectionCache:
    """Connection caching system for cloud services and databases"""

    def __init__(self, max_connections: int = 5, connection_timeout: float = 300.0):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self._connections: Dict[str, CachedConnection] = {}
        self._lock = threading.RLock()

    def get_connection(self, key: str, factory: Callable[[], Any]) -> Any:
        """Get or create a cached connection"""
        with self._lock:
            current_time = time.time()

            # Check if we have a valid cached connection
            if key in self._connections:
                cached = self._connections[key]
                if not cached.expired(current_time) and cached.is_valid():
                    cached.last_used = current_time
                    return cached.connection

                # Remove expired connection
                try:
                    cached.close()
                except Exception:
                    pass
                del self._connections[key]

            # Check connection limit
            if len(self._connections) >= self.max_connections:
                self._evict_oldest()

            # Create new connection
            try:
                connection = factory()
                cached = CachedConnection(connection, current_time, self.connection_timeout)
                self._connections[key] = cached
                return connection
            except Exception as e:
                logging.error(f"Failed to create connection for {key}: {e}")
                raise

    def invalidate_connection(self, key: str):
        """Invalidate a cached connection"""
        with self._lock:
            if key in self._connections:
                try:
                    self._connections[key].close()
                except Exception:
                    pass
                del self._connections[key]

    def _evict_oldest(self):
        """Evict the oldest unused connection"""
        oldest_key = None
        oldest_time = time.time()

        for key, cached in self._connections.items():
            if cached.last_used < oldest_time:
                oldest_time = cached.last_used
                oldest_key = key

        if oldest_key:
            try:
                self._connections[oldest_key].close()
            except Exception:
                pass
            del self._connections[oldest_key]

    def cleanup(self):
        """Cleanup all cached connections"""
        with self._lock:
            for cached in self._connections.values():
                try:
                    cached.close()
                except Exception:
                    pass
            self._connections.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get connection cache statistics"""
        with self._lock:
            return {
                "cached_connections": len(self._connections),
                "max_connections": self.max_connections,
                "connection_keys": list(self._connections.keys())
            }


class CachedConnection:
    """Wrapper for cached connections"""

    def __init__(self, connection: Any, created_at: float, timeout: float):
        self.connection = connection
        self.created_at = created_at
        self.last_used = created_at
        self.timeout = timeout

    def expired(self, current_time: float) -> bool:
        """Check if connection has expired"""
        return (current_time - self.last_used) > self.timeout

    def is_valid(self) -> bool:
        """Check if connection is still valid"""
        try:
            # Try to ping or check connection health
            if hasattr(self.connection, 'ping'):
                self.connection.ping()
            elif hasattr(self.connection, 'is_connected'):
                return self.connection.is_connected()
            elif hasattr(self.connection, 'closed'):
                return not self.connection.closed
            return True
        except Exception:
            return False

    def close(self):
        """Close the connection"""
        try:
            if hasattr(self.connection, 'close'):
                self.connection.close()
            elif hasattr(self.connection, 'disconnect'):
                self.connection.disconnect()
        except Exception:
            pass


class LazyLoader:
    """Intelligent lazy loading system for expensive resources"""

    def __init__(self):
        self._loaded_modules: Dict[str, Any] = {}
        self._loading_functions: Dict[str, Callable[[], Any]] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._lock = threading.RLock()

    def register_loader(self, name: str, loader_func: Callable[[], Any], dependencies: Optional[List[str]] = None):
        """Register a lazy loader function"""
        with self._lock:
            self._loading_functions[name] = loader_func
            if dependencies:
                self._dependencies[name] = dependencies

    def get(self, name: str) -> Any:
        """Get a resource, loading it if necessary"""
        with self._lock:
            if name in self._loaded_modules:
                return self._loaded_modules[name]

            # Check dependencies
            if name in self._dependencies:
                for dep in self._dependencies[name]:
                    if dep not in self._loaded_modules:
                        self.get(dep)  # Load dependency first

            # Load the resource
            if name in self._loading_functions:
                try:
                    resource = self._loading_functions[name]()
                    self._loaded_modules[name] = resource
                    logging.debug(f"Lazy loaded resource: {name}")
                    return resource
                except Exception as e:
                    logging.error(f"Failed to lazy load {name}: {e}")
                    raise
            else:
                raise KeyError(f"No loader registered for {name}")

    def is_loaded(self, name: str) -> bool:
        """Check if a resource is loaded"""
        with self._lock:
            return name in self._loaded_modules

    def preload(self, names: List[str]):
        """Preload multiple resources in background"""
        def preload_worker():
            for name in names:
                try:
                    self.get(name)
                except Exception as e:
                    logging.warning(f"Failed to preload {name}: {e}")

        thread = threading.Thread(target=preload_worker, daemon=True)
        thread.start()

    def unload(self, name: str):
        """Unload a resource"""
        with self._lock:
            if name in self._loaded_modules:
                resource = self._loaded_modules[name]
                try:
                    if hasattr(resource, 'cleanup'):
                        resource.cleanup()
                    elif hasattr(resource, 'close'):
                        resource.close()
                except Exception as e:
                    logging.warning(f"Error cleaning up {name}: {e}")
                del self._loaded_modules[name]


class MemoryManager:
    """Memory management and cleanup system"""

    def __init__(self, memory_threshold_mb: int = 500, cleanup_interval: float = 60.0):
        self.memory_threshold_mb = memory_threshold_mb
        self.cleanup_interval = cleanup_interval
        self._weak_refs: WeakSet = WeakSet()
        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._lock = threading.RLock()

        # Start memory monitoring thread
        self._monitor_thread = threading.Thread(target=self._memory_monitor, daemon=True)
        self._monitor_thread.start()

    def register_object(self, obj: Any):
        """Register an object for memory tracking"""
        with self._lock:
            self._weak_refs.add(ref(obj, self._object_cleanup_callback))

    def add_cleanup_callback(self, callback: Callable[[], None]):
        """Add a cleanup callback"""
        with self._lock:
            self._cleanup_callbacks.append(callback)

    def force_cleanup(self):
        """Force garbage collection and cleanup"""
        with self._lock:
            # Run cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logging.warning(f"Cleanup callback failed: {e}")

            # Force garbage collection
            collected = gc.collect()
            logging.debug(f"Garbage collection freed {collected} objects")

    def _object_cleanup_callback(self, weak_ref):
        """Callback when a tracked object is garbage collected"""
        with self._lock:
            try:
                self._weak_refs.remove(weak_ref)
            except KeyError:
                pass

    def _memory_monitor(self):
        """Monitor memory usage and trigger cleanup when needed"""
        while True:
            try:
                time.sleep(self.cleanup_interval)

                if HAS_PSUTIL:
                    # Check memory usage
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024

                    if memory_mb > self.memory_threshold_mb:
                        logging.warning(f"Memory usage high: {memory_mb:.1f}MB, triggering cleanup")
                        self.force_cleanup()

            except Exception as e:
                logging.error(f"Error in memory monitor: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            stats = {
                "tracked_objects": len(self._weak_refs),
                "cleanup_callbacks": len(self._cleanup_callbacks),
                "memory_threshold_mb": self.memory_threshold_mb
            }

            if HAS_PSUTIL:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                stats.update({
                    "rss_mb": memory_info.rss / 1024 / 1024,
                    "vms_mb": memory_info.vms / 1024 / 1024
                })

            return stats
        except Exception:
            return {"error": "Could not get memory stats"}


# Register lazy loaders for core components
resource_pool_manager = LazyLoader()

# Register core lazy loaders
resource_pool_manager.register_loader("config_manager", lambda: __import__('config').config_manager)
resource_pool_manager.register_loader("audio_analyzer", lambda: __import__('audio_analysis').audio_analyzer)
resource_pool_manager.register_loader("job_queue", lambda: __import__('job_queue').job_queue)
resource_pool_manager.register_loader("cloud_manager", lambda: __import__('cloud_integration').get_cloud_manager())
resource_pool_manager.register_loader("distributed_processor", lambda: __import__('cloud_integration').get_distributed_processor())
resource_pool_manager.register_loader("storage_manager", lambda: __import__('cloud_integration').get_storage_manager())
resource_pool_manager.register_loader("multi_stream_processor", lambda: __import__('multi_stream').multi_stream_processor)
resource_pool_manager.register_loader("selective_channel_processor", lambda: __import__('multi_stream').selective_channel_processor)

connection_cache = ConnectionCache()
memory_manager = MemoryManager()

# Convenience functions
def get_resource_pool(name: str) -> ResourcePool[Any]:
    """Get or create a named resource pool"""
    return resource_pool_manager.get(f"pool_{name}")

def get_connection(key: str, factory: Callable[[], Any]) -> Any:
    """Get a cached connection"""
    return connection_cache.get_connection(key, factory)

def lazy_load(name: str) -> Any:
    """Lazy load a resource"""
    return resource_pool_manager.get(name)

# Cleanup on exit
import atexit
atexit.register(connection_cache.cleanup)
atexit.register(memory_manager.force_cleanup)