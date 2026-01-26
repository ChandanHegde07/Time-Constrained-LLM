import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue
from config import get_config
from core.timer import get_timer_manager, TimerContextManager
from core.llm import get_llm_manager, LLMResponse, LLMResponseStatus

class TaskPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(order=True)
class QueuedTask:
    priority: int
    timestamp: float
    task_id: int
    task: 'Task' = field(compare=False)


@dataclass
class Task:
    task_id: int
    task_type: str
    prompt: str
    time_limit: float
    priority: TaskPriority = TaskPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingResult:
    success: bool
    task: Task
    assigned_to: str
    route_info: Dict[str, Any]
    timestamp: float


class RouteStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"
    LOAD_BALANCE = "load_balance"
    LEAST_CONNECTION = "least_connection"
    PERFORMANCE_BASED = "performance_based"

class TaskRouter:
    def __init__(self, config=None):
        self.config = config or get_config()
        self._routing_strategy = RouteStrategy.PRIORITY
        self._task_queue = PriorityQueue()
        self._task_counter = 0
        self._llm_manager = get_llm_manager()
        self._timer_manager = get_timer_manager()
        self._workers = []
        self._worker_lock = threading.Lock()
        self._stats = {
            "tasks_routed": 0,
            "tasks_completed": 0,
            "tasks_timed_out": 0,
            "tasks_failed": 0,
            "avg_routing_time": 0.0,
            "queue_wait_time": []
        }
        self._stats_lock = threading.Lock()

    def configure_strategy(self, strategy: RouteStrategy) -> None:
        self._routing_strategy = strategy

    def add_task(
        self,
        task_type: str,
        prompt: str,
        time_limit: float,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        self._task_counter += 1
        task = Task(
            task_id=self._task_counter,
            task_type=task_type,
            prompt=prompt,
            time_limit=time_limit,
            priority=priority,
            metadata=metadata or {}
        )

        queued_task = QueuedTask(
            priority=priority.value,
            timestamp=time.time(),
            task_id=task.task_id,
            task=task
        )

        self._task_queue.put(queued_task)

        self._update_stat("tasks_routed")

        return task.task_id

    def route_task(self) -> Optional[RoutingResult]:
        if self._task_queue.empty():
            return None

        start_time = time.time()

        try:
            queued_task = self._task_queue.get(timeout=0.1)
        except:
            return None

        queue_time = time.time() - queued_task.timestamp
        self._update_stat("queue_wait_time", queue_time)

        # Select appropriate route based on strategy
        route = self._select_route(queued_task.task)

        routing_time = time.time() - start_time

        self._update_stat("avg_routing_time", routing_time)

        return RoutingResult(
            success=True,
            task=queued_task.task,
            assigned_to=route,
            route_info={
                "queue_time": queue_time,
                "routing_time": routing_time,
                "strategy": self._routing_strategy.value
            },
            timestamp=time.time()
        )

    def _select_route(self, task: Task) -> str:
        if self._routing_strategy == RouteStrategy.PRIORITY:
            return self._select_priority_route(task)
        elif self._routing_strategy == RouteStrategy.LOAD_BALANCE:
            return self._select_load_balance_route(task)
        elif self._routing_strategy == RouteStrategy.LEAST_CONNECTION:
            return self._select_least_connection_route(task)
        elif self._routing_strategy == RouteStrategy.PERFORMANCE_BASED:
            return self._select_performance_based_route(task)
        else:
            return self._select_round_robin_route(task)

    def _select_priority_route(self, task: Task) -> str:
        if task.priority == TaskPriority.CRITICAL:
            return "high_priority_worker"
        elif task.priority == TaskPriority.HIGH:
            return "priority_worker"
        else:
            return "default_worker"

    def _select_round_robin_route(self, task: Task) -> str:
        workers = ["worker_1", "worker_2", "worker_3"]
        return workers[self._stats["tasks_routed"] % len(workers)]

    def _select_load_balance_route(self, task: Task) -> str:
        load_info = self._get_worker_load()
        if not load_info:
            return "default_worker"

        return min(load_info.items(), key=lambda x: x[1])[0]

    def _select_least_connection_route(self, task: Task) -> str:
        connection_info = self._get_worker_connections()
        if not connection_info:
            return "default_worker"

        return min(connection_info.items(), key=lambda x: x[1])[0]

    def _select_performance_based_route(self, task: Task) -> str:
        performance_info = self._get_worker_performance()
        if not performance_info:
            return "default_worker"

        return max(performance_info.items(), key=lambda x: x[1])[0]

    def _get_worker_load(self) -> Dict[str, float]:
        return {
            "worker_1": 0.4,
            "worker_2": 0.6,
            "worker_3": 0.3
        }

    def _get_worker_connections(self) -> Dict[str, int]:
        return {
            "worker_1": 3,
            "worker_2": 5,
            "worker_3": 2
        }

    def _get_worker_performance(self) -> Dict[str, float]:
        return {
            "worker_1": 0.95,
            "worker_2": 0.98,
            "worker_3": 0.92
        }

    def process_routed_task(self, routing_result: RoutingResult) -> LLMResponse:
        task = routing_result.task

        try:
            with TimerContextManager(task.time_limit) as timer:
                response = self._llm_manager.generate_response(
                    prompt=task.prompt,
                    time_limit=task.time_limit,
                    task_type=task.task_type
                )

                if timer.is_timeout():
                    response.status = LLMResponseStatus.TIMED_OUT
                else:
                    self._update_stat("tasks_completed")

        except Exception as e:
            self._update_stat("tasks_failed")
            raise e

        if response.status == LLMResponseStatus.TIMED_OUT:
            self._update_stat("tasks_timed_out")
        elif response.status == LLMResponseStatus.ERROR:
            self._update_stat("tasks_failed")

        return response

    def _update_stat(self, stat_name: str, value: Optional[float] = None) -> None:
        with self._stats_lock:
            if stat_name in ["tasks_routed", "tasks_completed", "tasks_timed_out", "tasks_failed"]:
                self._stats[stat_name] += 1
            elif stat_name == "avg_routing_time" and value is not None:
                current = self._stats["avg_routing_time"]
                count = self._stats["tasks_routed"]
                self._stats["avg_routing_time"] = ((current * (count - 1)) + value) / count
            elif stat_name == "queue_wait_time" and value is not None:
                self._stats["queue_wait_time"].append(value)

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
            if stats["queue_wait_time"]:
                stats["avg_queue_time"] = sum(stats["queue_wait_time"]) / len(stats["queue_wait_time"])
                stats["queue_wait_time"] = len(stats["queue_wait_time"])
            return stats

    def get_queue_size(self) -> int:
        return self._task_queue.qsize()

    def is_queue_empty(self) -> bool:
        return self._task_queue.empty()

    def get_task_info(self, task_id: int) -> Optional[Dict]:
        return None  


class RouterFactory:
    @classmethod
    def create_router(cls, strategy: Optional[RouteStrategy] = None, config=None) -> TaskRouter:
        router = TaskRouter(config)
        if strategy:
            router.configure_strategy(strategy)
        return router


class RoutingManager:
    _instance = None
    _routers = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._routers = {}
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._config = get_config()
            self._main_router = RouterFactory.create_router()

    def get_router(self, name: str = "main") -> TaskRouter:
        if name not in self._routers:
            self._routers[name] = RouterFactory.create_router()
        return self._routers[name]

    def get_default_router(self) -> TaskRouter:
        return self._main_router

    def set_routing_strategy(self, strategy: RouteStrategy, router_name: str = "main") -> None:
        router = self.get_router(router_name)
        router.configure_strategy(strategy)

    def add_task(
        self,
        task_type: str,
        prompt: str,
        time_limit: float,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        router_name: str = "main"
    ) -> int:
        router = self.get_router(router_name)
        return router.add_task(
            task_type, prompt, time_limit, priority, metadata
        )

    def process_next_task(self, router_name: str = "main") -> Optional[LLMResponse]:
        router = self.get_router(router_name)
        routing_result = router.route_task()
        if routing_result:
            return router.process_routed_task(routing_result)
        return None

    def get_router_stats(self, router_name: str = "main") -> Dict:
        router = self.get_router(router_name)
        return router.get_stats()

    def get_router_queue_size(self, router_name: str = "main") -> int:
        router = self.get_router(router_name)
        return router.get_queue_size()

# Global routing manager instance
_routing_manager = RoutingManager()

def get_routing_manager() -> RoutingManager:
    return _routing_manager

class TaskOrchestrator:
    def __init__(self):
        self._routing_manager = get_routing_manager()
        self._llm_manager = get_llm_manager()
        self._timer_manager = get_timer_manager()
        self._config = get_config()
        self._running = False
        self._workers = []
        self._worker_lock = threading.Lock()

    def start(self, worker_count: int = 3) -> None:
        if self._running:
            return

        self._running = True

        # Create worker threads
        for i in range(worker_count):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self._workers.append(worker)

    def stop(self) -> None:
        self._running = False
        for worker in self._workers:
            worker.join(timeout=1.0)

    def _worker_loop(self, worker_id: int) -> None:
        while self._running:
            try:
                response = self._routing_manager.process_next_task()
                if response:
                    self._handle_response(response, worker_id)
                else:
                    time.sleep(0.1)
            except Exception as e:
                self._handle_error(e, worker_id)

    def _handle_response(self, response: LLMResponse, worker_id: int) -> None:
        pass

    def _handle_error(self, error: Exception, worker_id: int) -> None:
        print(f"Worker {worker_id} error: {error}")

    def submit_task(
        self,
        task_type: str,
        prompt: str,
        time_limit: float,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        return self._routing_manager.add_task(
            task_type, prompt, time_limit, priority, metadata
        )

    def get_status(self) -> Dict:
        return {
            "running": self._running,
            "worker_count": len(self._workers),
            "active_workers": sum(1 for w in self._workers if w.is_alive()),
            "queue_size": self._routing_manager.get_router_queue_size(),
            "stats": self._routing_manager.get_router_stats()
        }

def get_task_orchestrator() -> TaskOrchestrator:
    return TaskOrchestrator()
