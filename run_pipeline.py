import time
import threading
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from config import get_config, get_config_value
from core.llm import get_llm_manager, LLMResponse, LLMResponseStatus
from core.timer import get_timer_manager, TimerContextManager
from core.router import get_routing_manager, TaskPriority
from core.evaluator import get_evaluator, EvaluationResult
from core.evaluator import AggregatedMetrics
from tasks import get_task_manager, generate_task

class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ExperimentConfig:
    experiment_name: str
    task_count: int = 10
    time_limits: List[float] = field(default_factory=lambda: [3.0, 5.0, 10.0])
    task_categories: List[str] = field(default_factory=lambda: ["reasoning", "creative", "qa"])
    difficulty_levels: List[int] = field(default_factory=lambda: [2, 3, 4])
    time_pressure_ratio: float = 0.5
    quality_scoring_enabled: bool = False
    statistical_analysis_enabled: bool = True
    output_formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    log_level: str = "INFO"
    timeout: float = 300.0


@dataclass
class ExperimentResult:
    experiment_name: str
    configuration: ExperimentConfig
    status: ExperimentStatus
    start_time: float
    end_time: float
    duration: float
    results: List[EvaluationResult] = field(default_factory=list)
    statistics: Optional[AggregatedMetrics] = None
    analysis: Dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._status = ExperimentStatus.PENDING
        self._results: List[EvaluationResult] = []
        self._start_time = 0.0
        self._end_time = 0.0
        self._progress = 0.0

        # Initialize components
        self._task_manager = get_task_manager()
        self._routing_manager = get_routing_manager()
        self._evaluator = get_evaluator()
        self._llm_manager = get_llm_manager()
        self._timer_manager = get_timer_manager()

        # Configure logging
        self._configure_logging()

    def _configure_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"outputs/experiment_{self.config.experiment_name}.log"),
                logging.StreamHandler()
            ]
        )
        self._logger = logging.getLogger(f"ExperimentRunner-{self.config.experiment_name}")

    def run(self) -> ExperimentResult:
        self._start_experiment()
        try:
            self._execute_experiment()
            self._finalize_experiment()
        except Exception as e:
            self._handle_error(e)

        return self._get_result()

    def _start_experiment(self) -> None:
        self._status = ExperimentStatus.RUNNING
        self._start_time = time.time()
        self._results.clear()

        self._logger.info("Experiment started: %s", self.config.experiment_name)
        self._logger.info("Configuration: %s", vars(self.config))

    def _execute_experiment(self) -> None:
        # Generate task batch
        batch = self._task_manager.generate_benchmark_batch(
            count=self.config.task_count,
            time_pressure_ratio=self.config.time_pressure_ratio
        )
        # Process tasks sequentially or in parallel
        if get_config_value("resources.max_concurrent_requests", 1) > 1:
            self._execute_parallel(batch)
        else:
            self._execute_sequential(batch)

    def _execute_sequential(self, batch: List[Tuple]) -> None:
        for i, (task, prompt, time_limit, is_pressure) in enumerate(batch, 1):
            if self._should_stop():
                break

            self._process_single_task(task, prompt, time_limit, i, is_pressure)

    def _execute_parallel(self, batch: List[Tuple]) -> None:
        max_concurrent = get_config_value("resources.max_concurrent_requests", 5)
        semaphore = threading.BoundedSemaphore(value=max_concurrent)

        def task_wrapper(i, task, prompt, time_limit, is_pressure):
            try:
                with semaphore:
                    self._process_single_task(task, prompt, time_limit, i, is_pressure)
            except Exception as e:
                self._logger.error("Error processing task %d: %s", i, e)

        threads = []
        for i, (task, prompt, time_limit, is_pressure) in enumerate(batch, 1):
            if self._should_stop():
                break

            thread = threading.Thread(
                target=task_wrapper,
                args=(i, task, prompt, time_limit, is_pressure),
                daemon=True
            )
            thread.start()
            threads.append(thread)
            time.sleep(0.1)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    def _process_single_task(
        self,
        task,
        prompt,
        time_limit: float,
        task_number: int,
        is_time_pressure: bool
    ) -> None:
        try:
            self._logger.info(
                "Processing task %d/%d: %s (%ss, %spressure)",
                task_number,
                self.config.task_count,
                task.name,
                time_limit,
                "with " if is_time_pressure else "without "
            )

            # Execute task with time constraint
            with TimerContextManager(time_limit) as timer:
                response = self._llm_manager.generate_response(
                    prompt=prompt,
                    time_limit=time_limit,
                    use_cache=True,
                    task_type=task.category.value
                )

                if timer.is_timeout():
                    response.status = LLMResponseStatus.TIMED_OUT

            # Evaluate response
            evaluation = self._evaluator.evaluate_response(
                response=response,
                task_type=task.category.value,
                time_limit=time_limit,
                task_id=task_number
            )

            self._results.append(evaluation)
            self._progress = task_number / self.config.task_count

            # Log task results
            self._logger.info(
                "Task %d completed: %s | Time: %.2fs | Quality: %.1f | Status: %s",
                task_number,
                task.category.value,
                response.time_elapsed,
                evaluation.metrics.get("quality_score", 0),
                response.status.value
            )

        except Exception as e:
            self._logger.error("Task %d failed: %s", task_number, e)

    def _should_stop(self) -> bool:
        if self._status != ExperimentStatus.RUNNING:
            return True

        runtime = time.time() - self._start_time
        if runtime >= self.config.timeout:
            self._logger.warning("Experiment timeout exceeded")
            return True

        return False

    def _handle_error(self, error: Exception) -> None:
        self._status = ExperimentStatus.ERROR
        self._end_time = time.time()
        self._logger.error("Experiment failed: %s", error)

    def _finalize_experiment(self) -> None:
        self._status = ExperimentStatus.COMPLETED
        self._end_time = time.time()

        # Generate reports and export results
        self._generate_reports()

        self._logger.info("Experiment completed successfully")
        self._logger.info("Duration: %.2fs", self._end_time - self._start_time)

    def _generate_reports(self) -> None:
        # Calculate statistics and analysis
        statistics = self._evaluator.get_statistics()
        analysis = self._evaluator.get_analysis()

        # Log key metrics
        self._logger.info("=== Experiment Statistics ===")
        self._logger.info("Total tasks: %d", statistics.total_tasks)
        self._logger.info("Completed tasks: %d (%.1f%%)",
                         statistics.completed_tasks,
                         statistics.completed_tasks / statistics.total_tasks * 100)
        self._logger.info("Timed out tasks: %d (%.1f%%)",
                         statistics.timed_out_tasks,
                         statistics.timed_out_tasks / statistics.total_tasks * 100)
        self._logger.info("Error tasks: %d (%.1f%%)",
                         statistics.error_tasks,
                         statistics.error_tasks / statistics.total_tasks * 100)
        self._logger.info("Average completion rate: %.1f%%",
                         statistics.avg_completion_rate * 100)
        self._logger.info("Average response time: %.2fs",
                         statistics.avg_response_time)
        self._logger.info("Average output length: %.1f characters",
                         statistics.avg_output_length)

        # Export results
        for format in self.config.output_formats:
            try:
                file_path = self._evaluator.save_results(format)
                self._logger.info("Results saved to: %s", file_path)
            except Exception as e:
                self._logger.error("Failed to save results in %s format: %s", format, e)

    def _get_result(self) -> ExperimentResult:
        return ExperimentResult(
            experiment_name=self.config.experiment_name,
            configuration=self.config,
            status=self._status,
            start_time=self._start_time,
            end_time=self._end_time,
            duration=self._end_time - self._start_time,
            results=self._results,
            statistics=self._evaluator.get_statistics(),
            analysis=self._evaluator.get_analysis()
        )

    def pause(self) -> None:
        if self._status == ExperimentStatus.RUNNING:
            self._status = ExperimentStatus.PAUSED
            self._logger.info("Experiment paused")

    def resume(self) -> None:
        if self._status == ExperimentStatus.PAUSED:
            self._status = ExperimentStatus.RUNNING
            self._logger.info("Experiment resumed")

    def stop(self) -> None:
        if self._status in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            self._status = ExperimentStatus.COMPLETED
            self._end_time = time.time()
            self._logger.info("Experiment stopped by user")


class ExperimentFactory:
    @classmethod
    def create_experiment(
        cls,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> ExperimentRunner:
        # Create base configuration
        base_config = ExperimentConfig(experiment_name=experiment_name)

        # Apply overrides from config dictionary
        if config:
            for key, value in config.items():
                if hasattr(base_config, key):
                    setattr(base_config, key, value)

        return ExperimentRunner(base_config)

    @classmethod
    def create_quick_test(cls, name: str = "quick_test") -> ExperimentRunner:
        config = ExperimentConfig(
            experiment_name=name,
            task_count=3,
            time_limits=[5.0, 10.0],
            task_categories=["reasoning", "qa"],
            difficulty_levels=[2, 3],
            time_pressure_ratio=0.5,
            quality_scoring_enabled=False,
            statistical_analysis_enabled=True,
            output_formats=["json"],
            log_level="INFO",
            timeout=300.0
        )
        return ExperimentRunner(config)

    @classmethod
    def create_comprehensive_test(cls, name: str = "comprehensive_test") -> ExperimentRunner:
        config = ExperimentConfig(
            experiment_name=name,
            task_count=50,
            time_limits=[3.0, 5.0, 10.0, 15.0, 30.0],
            task_categories=["reasoning", "creative", "qa", "analytical", "problem_solving"],
            difficulty_levels=[1, 2, 3, 4, 5],
            time_pressure_ratio=0.5,
            quality_scoring_enabled=True,
            statistical_analysis_enabled=True,
            output_formats=["json", "csv"],
            log_level="DEBUG",
            timeout=600.0
        )
        return ExperimentRunner(config)


class PipelineController:
    def __init__(self):
        self._active_experiments = {}
        self._experiment_lock = threading.Lock()
        self._config = get_config()

    def create_experiment(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> ExperimentRunner:
        experiment = ExperimentFactory.create_experiment(name, config)

        with self._experiment_lock:
            self._active_experiments[name] = experiment

        return experiment

    def run_experiment(self, name: str, config: Optional[Dict[str, Any]] = None) -> ExperimentResult:
        experiment = self.create_experiment(name, config)
        return experiment.run()

    def get_experiment(self, name: str) -> Optional[ExperimentRunner]:
        with self._experiment_lock:
            return self._active_experiments.get(name)

    def get_active_experiments(self) -> List[str]:
        with self._experiment_lock:
            return list(self._active_experiments.keys())

    def remove_experiment(self, name: str) -> bool:
        with self._experiment_lock:
            if name in self._active_experiments:
                del self._active_experiments[name]
                return True
            return False

# Global pipeline controller instance
_pipeline_controller = PipelineController()

def get_pipeline_controller() -> PipelineController:
    return _pipeline_controller

# Main entry point
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Time Constrained LLM System")

    parser.add_argument("--name", default="default_experiment",
                       help="Experiment name")
    parser.add_argument("--tasks", type=int, default=10,
                       help="Number of tasks")
    parser.add_argument("--time-limits", nargs='+', type=float,
                       default=[3.0, 5.0, 10.0],
                       help="Time limits for tasks")
    parser.add_argument("--categories", nargs='+',
                       default=["reasoning", "creative", "qa"],
                       help="Task categories")
    parser.add_argument("--difficulty", nargs='+', type=int,
                       default=[2, 3, 4],
                       help="Difficulty levels (1-5)")
    parser.add_argument("--pressure-ratio", type=float, default=0.5,
                       help="Time pressure ratio (0-1)")
    parser.add_argument("--no-quality-scoring", action="store_false",
                       dest="quality_scoring",
                       help="Disable quality scoring")
    parser.add_argument("--output-formats", nargs='+',
                       default=["json", "csv"],
                       help="Output formats")
    parser.add_argument("--log-level", default="INFO",
                       help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick test configuration")

    args = parser.parse_args()

    print(f"Time Constrained LLM System - Starting experiment '{args.name}'")

    # Create and run experiment
    try:
        if args.quick_test:
            experiment = ExperimentFactory.create_quick_test(args.name)
        else:
            config = ExperimentConfig(
                experiment_name=args.name,
                task_count=args.tasks,
                time_limits=args.time_limits,
                task_categories=args.categories,
                difficulty_levels=args.difficulty,
                time_pressure_ratio=args.pressure_ratio,
                quality_scoring_enabled=args.quality_scoring,
                statistical_analysis_enabled=True,
                output_formats=args.output_formats,
                log_level=args.log_level,
                timeout=300.0
            )
            experiment = ExperimentRunner(config)

        result = experiment.run()

        print("\nExperiment completed successfully!")
        print(f"Duration: {result.duration:.2f} seconds")
        print(f"Results saved to: outputs/ directory")

    except Exception as e:
        print(f"\nError: {e}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()