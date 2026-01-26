import time
import threading
import queue
from typing import Callable, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from config import get_config


class TimerState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    PAUSED = "paused"


@dataclass
class TimingMetrics:
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    time_remaining: float = 0.0
    callbacks_executed: int = 0
    deadline_exceeded: bool = False
    precision: float = 0.001
    metrics: dict = field(default_factory=dict)


@dataclass
class TimerEvent:
    event_type: str
    timestamp: float
    timer_state: TimerState
    metrics: TimingMetrics
    data: dict = field(default_factory=dict)


class PrecisionTimer:
    def __init__(
        self,
        time_limit: float,
        precision: float = 0.001,
        config=None
    ):
        self.config = config or get_config().timer
        self.time_limit = time_limit
        self.precision = precision
        self.state = TimerState.IDLE
        self._start_time = 0.0
        self._end_time = 0.0
        self._elapsed = 0.0
        self._remaining = time_limit
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._timer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._events = queue.Queue()
        self._callbacks = []
        self._metrics = TimingMetrics(precision=precision)

    def start(self) -> None:
        with self._lock:
            if self.state == TimerState.RUNNING:
                return

            self.state = TimerState.RUNNING
            self._start_time = time.perf_counter()
            self._end_time = 0.0
            self._elapsed = 0.0
            self._remaining = self.time_limit
            self._stop_event.clear()
            self._events.queue.clear()

            self._metrics = TimingMetrics(
                start_time=self._start_time,
                precision=self.precision
            )

            # Start timer thread
            self._timer_thread = threading.Thread(
                target=self._timer_loop,
                daemon=True
            )
            self._timer_thread.start()

            self._add_event("timer_started")

    def _timer_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                time.sleep(self.precision)
                self._update_metrics()

                if self._remaining <= 0:
                    self._handle_timeout()
                    break

                # Check for early termination conditions
                if self._should_stop_early():
                    break

        except Exception as e:
            self._add_event("timer_error", error=str(e))

    def _update_metrics(self) -> None:
        with self._lock:
            current_time = time.perf_counter()
            self._elapsed = current_time - self._start_time
            self._remaining = self.time_limit - self._elapsed

            self._metrics.end_time = current_time
            self._metrics.duration = self._elapsed
            self._metrics.time_remaining = self._remaining

    def _handle_timeout(self) -> None:
        self._stop_event.set()
        with self._lock:
            self.state = TimerState.TIMED_OUT
            self._metrics.deadline_exceeded = True
            self._add_event("time_limit_exceeded")
            self._execute_callbacks()

    def _should_stop_early(self) -> bool:
        return False  # Implement custom early stop logic here

    def stop(self) -> None:
        with self._lock:
            if self.state != TimerState.RUNNING:
                return

            self._stop_event.set()
            self.state = TimerState.COMPLETED
            self._end_time = time.perf_counter()
            self._elapsed = self._end_time - self._start_time
            self._remaining = self.time_limit - self._elapsed

            self._metrics.end_time = self._end_time
            self._metrics.duration = self._elapsed
            self._metrics.time_remaining = self._remaining

            self._add_event("timer_stopped")

    def pause(self) -> None:
        with self._lock:
            if self.state != TimerState.RUNNING:
                return

            self.state = TimerState.PAUSED
            self._stop_event.set()
            self._update_metrics()
            self._add_event("timer_paused")

    def resume(self) -> None:
        with self._lock:
            if self.state != TimerState.PAUSED:
                return

            self.state = TimerState.RUNNING
            self._start_time += time.perf_counter() - self._end_time
            self._stop_event.clear()

            self._timer_thread = threading.Thread(
                target=self._timer_loop,
                daemon=True
            )
            self._timer_thread.start()

            self._add_event("timer_resumed")

    def is_timeout(self) -> bool:
        return self.state == TimerState.TIMED_OUT

    def is_running(self) -> bool:
        return self.state == TimerState.RUNNING

    def get_time_remaining(self) -> float:
        with self._lock:
            if self.state == TimerState.IDLE:
                return self.time_limit
            elif self.state == TimerState.PAUSED:
                return self._remaining
            elif self.state == TimerState.RUNNING:
                current_time = time.perf_counter()
                return max(0.0, self.time_limit - (current_time - self._start_time))
            else:
                return 0.0

    def get_elapsed_time(self) -> float:
        with self._lock:
            if self.state == TimerState.IDLE:
                return 0.0
            elif self.state == TimerState.PAUSED:
                return self._elapsed
            elif self.state == TimerState.RUNNING:
                return time.perf_counter() - self._start_time
            else:
                return self._elapsed

    def get_metrics(self) -> TimingMetrics:
        with self._lock:
            self._update_metrics()
            return self._metrics

    def add_callback(
        self,
        callback: Callable[[TimerEvent], None],
        trigger_condition: Optional[Callable[[TimingMetrics], bool]] = None
    ) -> None:
        with self._lock:
            self._callbacks.append((callback, trigger_condition))

    def _execute_callbacks(self) -> None:
        for callback, condition in self._callbacks:
            try:
                if condition is None or condition(self._metrics):
                    callback(self._get_current_event())
                    self._metrics.callbacks_executed += 1
            except Exception as e:
                self._add_event("callback_error", error=str(e))

    def _add_event(self, event_type: str, **kwargs) -> None:
        event = TimerEvent(
            event_type=event_type,
            timestamp=time.perf_counter(),
            timer_state=self.state,
            metrics=self._metrics,
            data=kwargs
        )
        self._events.put(event)

    def _get_current_event(self) -> TimerEvent:
        return TimerEvent(
            event_type="timer_update",
            timestamp=time.perf_counter(),
            timer_state=self.state,
            metrics=self._metrics
        )

    def get_events(self) -> list:
        events = []
        while not self._events.empty():
            events.append(self._events.get())
        return events

    def wait(self, timeout: Optional[float] = None) -> bool:
        start_time = time.perf_counter()

        while self.is_running():
            current_time = time.perf_counter()
            if timeout is not None and (current_time - start_time) >= timeout:
                return False

            time.sleep(min(self.precision, 0.1))
        return True


class TimerFactory:
    @classmethod
    def create_timer(
        cls,
        time_limit: float,
        precision: Optional[float] = None,
        config=None
    ) -> PrecisionTimer:
        if precision is None:
            precision = (config or get_config().timer).precision

        return PrecisionTimer(
            time_limit=time_limit,
            precision=precision,
            config=config
        )

class TimerManager:
    _instance = None
    _timers = {}
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._timers = {}
            cls._next_id = 1
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._config = get_config().timer
            self._timer_lock = threading.Lock()
            self._timer_count = 0

    def create_timer(
        self,
        time_limit: float,
        precision: Optional[float] = None
    ) -> Tuple[int, PrecisionTimer]:
        timer = TimerFactory.create_timer(time_limit, precision)

        with self._timer_lock:
            timer_id = self._next_id
            self._next_id += 1
            self._timers[timer_id] = timer
            self._timer_count += 1

        return timer_id, timer

    def get_timer(self, timer_id: int) -> Optional[PrecisionTimer]:
        with self._timer_lock:
            return self._timers.get(timer_id)

    def remove_timer(self, timer_id: int) -> bool:
        with self._timer_lock:
            if timer_id in self._timers:
                del self._timers[timer_id]
                self._timer_count -= 1
                return True
            return False

    def get_all_timers(self) -> dict:
        with self._timer_lock:
            return dict(self._timers)

    def get_active_timers(self) -> dict:
        with self._timer_lock:
            return {
                timer_id: timer
                for timer_id, timer in self._timers.items()
                if timer.is_running()
            }

    def get_timer_count(self) -> int:
        with self._timer_lock:
            return self._timer_count

    def get_active_count(self) -> int:
        with self._timer_lock:
            return sum(1 for timer in self._timers.values() if timer.is_running())

    def stop_all_timers(self) -> None:
        with self._timer_lock:
            for timer in self._timers.values():
                if timer.is_running():
                    timer.stop()

    def clear_all_timers(self) -> None:
        self.stop_all_timers()
        with self._timer_lock:
            self._timers.clear()
            self._timer_count = 0


# Global timer manager instance
_timer_manager = TimerManager()


def get_timer_manager() -> TimerManager:
    return _timer_manager


class TimerContextManager:
    def __init__(self, time_limit: float, precision: float = 0.001):
        self.time_limit = time_limit
        self.precision = precision
        self._timer = None

    def __enter__(self):
        self._timer = PrecisionTimer(self.time_limit, self.precision)
        self._timer.start()
        return self._timer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._timer and self._timer.is_running():
            self._timer.stop()
        return False


# Decorator for timed functions
def timed(time_limit: float, precision: float = 0.001):
    def decorator(func):
        def wrapper(*args, **kwargs):
            timer = PrecisionTimer(time_limit, precision)
            timer.start()

            result = None
            exception = None

            def target():
                nonlocal result, exception
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    exception = e
                finally:
                    timer.stop()

            thread = threading.Thread(target=target, daemon=True)
            thread.start()

            if timer.wait(timeout=time_limit + 0.1):
                if exception:
                    raise exception
                return result
            else:
                raise TimeoutError(f"Function '{func.__name__}' timed out after {time_limit} seconds")

        return wrapper

    return decorator
