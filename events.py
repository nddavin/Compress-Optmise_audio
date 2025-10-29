"""
Centralized Event System for Cross-Module Communication

This module provides a pub-sub event system that enables loose coupling between
modules while allowing them to communicate through events.
"""

import threading
import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time


class EventPriority(Enum):
    """Priority levels for event handling"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """Represents an event in the system"""
    name: str
    data: Dict[str, Any]
    source: str
    timestamp: Optional[float] = None
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class EventBus:
    """
    Centralized event bus for pub-sub communication between modules.

    Features:
    - Thread-safe event publishing and subscription
    - Priority-based event handling
    - Event filtering and correlation
    - Asynchronous event processing
    - Event history and replay capabilities
    """

    def __init__(self, max_history: int = 1000):
        self._subscribers: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self._event_history: List[Event] = []
        self._max_history = max_history
        self._running = True
        self._event_queue: List[Event] = []
        self._queue_lock = threading.Lock()
        self._worker_thread = threading.Thread(target=self._process_events, daemon=True)
        self._worker_thread.start()

    def subscribe(self, event_name: str, callback: Callable[[Event], None],
                  priority: EventPriority = EventPriority.NORMAL,
                  filter_func: Optional[Callable[[Event], bool]] = None) -> str:
        """
        Subscribe to an event with optional filtering.

        Args:
            event_name: Name of the event to subscribe to (supports wildcards with *)
            callback: Function to call when event is published
            priority: Priority level for event handling
            filter_func: Optional function to filter events

        Returns:
            Subscription ID for unsubscribing
        """
        import uuid
        subscription_id = str(uuid.uuid4())

        with self._lock:
            if event_name not in self._subscribers:
                self._subscribers[event_name] = []

            self._subscribers[event_name].append({
                'id': subscription_id,
                'callback': callback,
                'priority': priority,
                'filter_func': filter_func
            })

            # Sort subscribers by priority (higher priority first)
            self._subscribers[event_name].sort(key=lambda x: x['priority'].value, reverse=True)

        logging.debug(f"Subscribed to event '{event_name}' with ID {subscription_id}")
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from an event using subscription ID"""
        with self._lock:
            for event_name, subscribers in self._subscribers.items():
                for subscriber in subscribers[:]:  # Copy to avoid modification during iteration
                    if subscriber['id'] == subscription_id:
                        subscribers.remove(subscriber)
                        logging.debug(f"Unsubscribed from event '{event_name}' with ID {subscription_id}")
                        return True
        return False

    def publish(self, event: Event, synchronous: bool = False) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
            synchronous: If True, process immediately; otherwise queue for async processing
        """
        if synchronous:
            self._process_event_immediately(event)
        else:
            with self._queue_lock:
                self._event_queue.append(event)

        # Add to history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

        logging.debug(f"Published event '{event.name}' from '{event.source}'")

    def publish_event(self, name: str, data: Dict[str, Any], source: str,
                     priority: EventPriority = EventPriority.NORMAL,
                     correlation_id: Optional[str] = None,
                     synchronous: bool = False) -> None:
        """Convenience method to publish an event"""
        event = Event(
            name=name,
            data=data,
            source=source,
            priority=priority,
            correlation_id=correlation_id
        )
        self.publish(event, synchronous)

    def _process_events(self):
        """Background worker to process queued events"""
        while self._running:
            events_to_process = []
            with self._queue_lock:
                if self._event_queue:
                    events_to_process = self._event_queue[:]
                    self._event_queue.clear()

            for event in events_to_process:
                try:
                    self._process_event_immediately(event)
                except Exception as e:
                    logging.error(f"Error processing event '{event.name}': {e}")

            time.sleep(0.01)  # Small delay to prevent busy waiting

    def _process_event_immediately(self, event: Event):
        """Process a single event immediately"""
        with self._lock:
            # Find matching subscribers (support wildcards)
            matching_subscribers = []
            for event_pattern, subscribers in self._subscribers.items():
                if self._matches_pattern(event.name, event_pattern):
                    matching_subscribers.extend(subscribers)

            # Sort by priority
            matching_subscribers.sort(key=lambda x: x['priority'].value, reverse=True)

        # Process subscribers
        for subscriber in matching_subscribers:
            try:
                # Apply filter if present
                if subscriber['filter_func'] and not subscriber['filter_func'](event):
                    continue

                # Call callback in a separate thread to prevent blocking
                threading.Thread(
                    target=self._safe_callback,
                    args=(subscriber['callback'], event),
                    daemon=True
                ).start()

            except Exception as e:
                logging.error(f"Error in event subscriber for '{event.name}': {e}")

    def _safe_callback(self, callback: Callable[[Event], None], event: Event):
        """Safely execute callback with error handling"""
        try:
            callback(event)
        except Exception as e:
            logging.error(f"Event callback failed for '{event.name}': {e}")

    def _matches_pattern(self, event_name: str, pattern: str) -> bool:
        """Check if event name matches pattern (supports * wildcards)"""
        if '*' not in pattern:
            return event_name == pattern

        # Simple wildcard matching
        import fnmatch
        return fnmatch.fnmatch(event_name, pattern)

    def get_event_history(self, event_name: Optional[str] = None,
                         limit: int = 100) -> List[Event]:
        """Get recent events from history"""
        with self._lock:
            history = self._event_history[-limit:]

            if event_name:
                history = [e for e in history if self._matches_pattern(e.name, event_name)]

            return history.copy()

    def clear_history(self):
        """Clear event history"""
        with self._lock:
            self._event_history.clear()

    def get_subscriber_count(self, event_name: Optional[str] = None) -> int:
        """Get number of subscribers for an event or total"""
        with self._lock:
            if event_name:
                return len(self._subscribers.get(event_name, []))
            else:
                return sum(len(subs) for subs in self._subscribers.values())

    def shutdown(self):
        """Shutdown the event bus"""
        self._running = False
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)


# Global event bus instance
event_bus = EventBus()


# Convenience functions for common event operations
def subscribe(event_name: str, callback: Callable[[Event], None],
             priority: EventPriority = EventPriority.NORMAL,
             filter_func: Optional[Callable[[Event], bool]] = None) -> str:
    """Subscribe to an event"""
    return event_bus.subscribe(event_name, callback, priority, filter_func)


def unsubscribe(subscription_id: str) -> bool:
    """Unsubscribe from an event"""
    return event_bus.unsubscribe(subscription_id)


def publish_event(name: str, data: Dict[str, Any], source: str,
                 priority: EventPriority = EventPriority.NORMAL,
                 correlation_id: Optional[str] = None) -> None:
    """Publish an event"""
    event_bus.publish_event(name, data, source, priority, correlation_id)


def get_event_history(event_name: Optional[str] = None, limit: int = 100) -> List[Event]:
    """Get event history"""
    return event_bus.get_event_history(event_name, limit)