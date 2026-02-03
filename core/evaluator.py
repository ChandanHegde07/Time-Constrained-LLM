import time
import json
import csv
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import scipy.stats as stats
from config import get_config
from core.llm import LLMResponse, LLMResponseStatus

class EvaluationMetric(Enum):
    COMPLETION_RATE = "completion_rate"
    TIME_TO_CUTOFF = "time_to_cutoff"
    OUTPUT_LENGTH = "output_length"
    QUALITY_SCORE = "quality_score"
    RESPONSE_TIME = "response_time"
    TOKEN_RATE = "token_rate"
    ERROR_RATE = "error_rate"
    PARTIAL_COMPLETION = "partial_completion"


@dataclass
class EvaluationResult:
    task_id: int
    task_type: str
    time_limit: float
    response: LLMResponse
    metrics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedMetrics:
    total_tasks: int
    completed_tasks: int
    timed_out_tasks: int
    error_tasks: int
    avg_completion_rate: float
    avg_response_time: float
    avg_output_length: float
    completion_rate_by_timelimit: Dict[float, float]
    response_time_distribution: Dict[float, float]
    quality_scores: Dict[str, float]
    statistical_summary: Dict[str, Any] = field(default_factory=dict)


class QualityScorer:
    def __init__(self, config=None):
        self.config = config or get_config().evaluation
        self._scorers = self._initialize_scorers()

    def _initialize_scorers(self) -> Dict[str, Callable]:
        return {
            "completeness": self._score_completeness,
            "relevance": self._score_relevance,
            "coherence": self._score_coherence,
            "accuracy": self._score_accuracy
        }

    def score_response(self, response: LLMResponse, task_type: str) -> float:
        if not self.config.quality_scoring_enabled or response.status == LLMResponseStatus.ERROR:
            return 0.0

        scores = []
        for name, scorer in self._scorers.items():
            try:
                scores.append(scorer(response, task_type))
            except Exception as e:
                print(f"Warning: Failed to score {name}: {e}")

        return sum(scores) / len(scores) * 100 if scores else 0.0

    def _score_completeness(self, response: LLMResponse, task_type: str) -> float:
        if response.status == LLMResponseStatus.PARTIAL:
            return 0.5
        return 1.0 if response.content.strip() else 0.0

    def _score_relevance(self, response: LLMResponse, task_type: str) -> float:
        return 0.8 if response.content.strip() else 0.0

    def _score_coherence(self, response: LLMResponse, task_type: str) -> float:
        return 0.7 if len(response.content.split()) > 5 else 0.3

    def _score_accuracy(self, response: LLMResponse, task_type: str) -> float:
        return 0.6  # Placeholder implementation


class MetricCalculator:
    def __init__(self, scorer: Optional[QualityScorer] = None, config=None):
        self.config = config or get_config().evaluation
        self._scorer = scorer or QualityScorer(config)

    def calculate_metrics(
        self,
        response: LLMResponse,
        task_type: str,
        time_limit: float
    ) -> Dict[str, float]:
        metrics = {}

        for metric in self.config.metrics:
            if metric == EvaluationMetric.COMPLETION_RATE.value:
                metrics[metric] = self._calculate_completion_rate(response, time_limit)
            elif metric == EvaluationMetric.TIME_TO_CUTOFF.value:
                metrics[metric] = self._calculate_time_to_cutoff(response)
            elif metric == EvaluationMetric.OUTPUT_LENGTH.value:
                metrics[metric] = self._calculate_output_length(response)
            elif metric == EvaluationMetric.QUALITY_SCORE.value:
                metrics[metric] = self._calculate_quality_score(response, task_type)
            elif metric == EvaluationMetric.RESPONSE_TIME.value:
                metrics[metric] = self._calculate_response_time(response)
            elif metric == EvaluationMetric.TOKEN_RATE.value:
                metrics[metric] = self._calculate_token_rate(response)
            elif metric == EvaluationMetric.ERROR_RATE.value:
                metrics[metric] = self._calculate_error_rate(response)
            elif metric == EvaluationMetric.PARTIAL_COMPLETION.value:
                metrics[metric] = self._calculate_partial_completion(response)

        return metrics

    def _calculate_completion_rate(self, response: LLMResponse, time_limit: float) -> float:
        if response.status == LLMResponseStatus.COMPLETED:
            return 1.0
        elif response.status == LLMResponseStatus.PARTIAL:
            return min(response.time_elapsed / time_limit, 0.9)
        else:
            return 0.0

    def _calculate_time_to_cutoff(self, response: LLMResponse) -> float:
        return response.time_elapsed

    def _calculate_output_length(self, response: LLMResponse) -> float:
        return len(response.content)

    def _calculate_quality_score(self, response: LLMResponse, task_type: str) -> float:
        return self._scorer.score_response(response, task_type)

    def _calculate_response_time(self, response: LLMResponse) -> float:
        return response.time_elapsed

    def _calculate_token_rate(self, response: LLMResponse) -> float:
        if response.time_elapsed > 0:
            return response.token_count / response.time_elapsed
        return 0.0

    def _calculate_error_rate(self, response: LLMResponse) -> float:
        return 1.0 if response.status == LLMResponseStatus.ERROR else 0.0

    def _calculate_partial_completion(self, response: LLMResponse) -> float:
        return 1.0 if response.status == LLMResponseStatus.PARTIAL else 0.0


class StatisticalAnalyzer:
    def __init__(self, config=None):
        self.config = config or get_config().evaluation
    
    def analyze_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        if not results:
            return {}

        analysis = {}

        # Calculate basic statistics
        analysis["basic"] = self._calculate_basic_statistics(results)

        # Calculate time limit-specific analysis
        analysis["time_limit_analysis"] = self._analyze_time_limit_effects(results)

        # Calculate task type analysis
        analysis["task_type_analysis"] = self._analyze_task_type_effects(results)

        # Calculate correlation analysis
        analysis["correlations"] = self._calculate_correlations(results)

        # Calculate distribution analysis
        analysis["distributions"] = self._analyze_distributions(results)

        # Calculate confidence intervals
        analysis["confidence_intervals"] = self._calculate_confidence_intervals(results)

        return analysis

    def _calculate_basic_statistics(self, results: List[EvaluationResult]) -> Dict:
        scores = [r.metrics.get("quality_score", 0) for r in results]
        times = [r.metrics.get("response_time", 0) for r in results]

        return {
            "count": len(results),
            "completion_rate": sum(1 for r in results if r.response.status == LLMResponseStatus.COMPLETED) / len(results),
            "avg_quality": np.mean(scores),
            "std_quality": np.std(scores),
            "min_quality": np.min(scores),
            "max_quality": np.max(scores),
            "avg_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "total_tokens": sum(r.response.token_count for r in results),
            "avg_tokens": np.mean([r.response.token_count for r in results])
        }

    def _analyze_time_limit_effects(self, results: List[EvaluationResult]) -> Dict:
        by_time = defaultdict(list)
        for result in results:
            by_time[result.time_limit].append(result)

        time_analysis = {}
        for time_limit, time_results in by_time.items():
            scores = [r.metrics["quality_score"] for r in results]
            completion_rates = [r.metrics["completion_rate"] for r in results]

            time_analysis[time_limit] = {
                "count": len(time_results),
                "avg_quality": np.mean(scores),
                "std_quality": np.std(scores),
                "completion_rate": np.mean(completion_rates),
                "avg_response_time": np.mean([r.response.time_elapsed for r in results])
            }

        return dict(time_analysis)

    def _analyze_task_type_effects(self, results: List[EvaluationResult]) -> Dict:
        by_task = defaultdict(list)
        for result in results:
            by_task[result.task_type].append(result)

        task_analysis = {}
        for task_type, task_results in by_task.items():
            scores = [r.metrics["quality_score"] for r in results]
            completion_rates = [r.metrics["completion_rate"] for r in results]

            task_analysis[task_type] = {
                "count": len(task_results),
                "avg_quality": np.mean(scores),
                "std_quality": np.std(scores),
                "completion_rate": np.mean(completion_rates),
                "avg_response_time": np.mean([r.response.time_elapsed for r in results])
            }

        return dict(task_analysis)

    def _calculate_correlations(self, results: List[EvaluationResult]) -> Dict:
        metrics = list(results[0].metrics.keys())
        correlations = {}

        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics[i+1:], i+1):
                values1 = [r.metrics[metric1] for r in results]
                values2 = [r.metrics[metric2] for r in results]
                correlation = stats.pearsonr(values1, values2)
                correlations[f"{metric1}-{metric2}"] = {
                    "correlation": correlation[0],
                    "p_value": correlation[1]
                }

        return correlations

    def _analyze_distributions(self, results: List[EvaluationResult]) -> Dict:
        distributions = {}

        for metric in results[0].metrics.keys():
            values = [r.metrics[metric] for r in results]
            quartiles = np.percentile(values, [25, 50, 75])
            kurtosis = stats.kurtosis(values)
            skewness = stats.skew(values)

            distributions[metric] = {
                "quartiles": {
                    "25th": quartiles[0],
                    "50th": quartiles[1],
                    "75th": quartiles[2]
                },
                "kurtosis": kurtosis,
                "skewness": skewness,
                "histogram": self._calculate_histogram(values)
            }

        return distributions

    def _calculate_histogram(self, values: List[float]) -> Dict:
        hist, bins = np.histogram(values, bins='auto')
        return {
            "bins": bins.tolist(),
            "counts": hist.tolist()
        }

    def _calculate_confidence_intervals(self, results: List[EvaluationResult]) -> Dict:
        intervals = {}

        for metric in results[0].metrics.keys():
            values = [r.metrics[metric] for r in results]
            mean = np.mean(values)
            std_err = stats.sem(values)
            interval = stats.t.interval(
                self.config.confidence_interval,
                len(values) - 1,
                loc=mean,
                scale=std_err
            )

            intervals[metric] = {
                "mean": mean,
                "confidence_interval": {
                    "lower": interval[0],
                    "upper": interval[1],
                    "level": self.config.confidence_interval
                },
                "sample_size": len(values)
            }

        return intervals


class ResultsManager:
    def __init__(self, config=None):
        self.config = config or get_config().output
        self._results = []
        self._results_lock = threading.Lock()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure required directories exist for output storage."""
        Path(self.config.output_directory).mkdir(exist_ok=True)

    def add_result(self, result: EvaluationResult) -> int:
        with self._results_lock:
            self._results.append(result)
            return len(self._results) - 1

    def get_all_results(self) -> List[EvaluationResult]:
        with self._results_lock:
            return list(self._results)

    def get_results_by_task_type(self, task_type: str) -> List[EvaluationResult]:
        with self._results_lock:
            return [r for r in self._results if r.task_type == task_type]

    def get_results_by_time_limit(self, time_limit: float) -> List[EvaluationResult]:
        with self._results_lock:
            return [r for r in self._results if r.time_limit == time_limit]

    def get_statistics(self) -> AggregatedMetrics:
        with self._results_lock:
            if not self._results:
                return AggregatedMetrics(
                    total_tasks=0,
                    completed_tasks=0,
                    timed_out_tasks=0,
                    error_tasks=0,
                    avg_completion_rate=0.0,
                    avg_response_time=0.0,
                    avg_output_length=0.0,
                    completion_rate_by_timelimit={},
                    response_time_distribution={},
                    quality_scores={}
                )

            completed = sum(1 for r in self._results if r.response.status == LLMResponseStatus.COMPLETED)
            timed_out = sum(1 for r in self._results if r.response.status == LLMResponseStatus.TIMED_OUT)
            errors = sum(1 for r in self._results if r.response.status == LLMResponseStatus.ERROR)

            by_time = defaultdict(list)
            for result in self._results:
                by_time[result.time_limit].append(result)

            completion_by_time = {
                time_limit: sum(1 for r in results if r.response.status == LLMResponseStatus.COMPLETED) / len(results)
                for time_limit, results in by_time.items()
            }

            return AggregatedMetrics(
                total_tasks=len(self._results),
                completed_tasks=completed,
                timed_out_tasks=timed_out,
                error_tasks=errors,
                avg_completion_rate=sum(r.metrics["completion_rate"] for r in self._results) / len(self._results),
                avg_response_time=sum(r.response.time_elapsed for r in self._results) / len(self._results),
                avg_output_length=sum(len(r.response.content) for r in self._results) / len(self._results),
                completion_rate_by_timelimit=dict(completion_by_time),
                response_time_distribution={},
                quality_scores={}
            )

    def save_results(self, format: str = "json") -> str:
        file_path = self._get_output_path(format)

        if format == "json":
            self._save_json(file_path)
        elif format == "csv":
            self._save_csv(file_path)

        return file_path

    def _get_output_path(self, format: str) -> str:
        return f"{self.config.output_directory}/{self.config.response_file}"

    def _save_json(self, file_path: str) -> None:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([self._result_to_dict(r) for r in self._results], f, ensure_ascii=False, indent=2)

    def _save_csv(self, file_path: str) -> None:
        csv_path = f"{self.config.output_directory}/{self.config.metrics_file}"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self._get_csv_fields())
            writer.writeheader()

            for result in self._results:
                writer.writerow(self._result_to_csv(result))

    def _get_csv_fields(self) -> List[str]:
        return [
            "task_id", "task_type", "time_limit", "response_status",
            "response_time", "completion_rate", "quality_score",
            "token_count", "token_rate", "output_length", "timestamp"
        ]

    def _result_to_dict(self, result: EvaluationResult) -> Dict:
        return {
            "task_id": result.task_id,
            "task_type": result.task_type,
            "time_limit": result.time_limit,
            "response": {
                "content": result.response.content,
                "status": result.response.status.value,
                "time_elapsed": result.response.time_elapsed,
                "token_count": result.response.token_count,
                "model": result.response.model,
                "temperature": result.response.temperature,
                "finish_reason": result.response.finish_reason,
                "partial_reason": result.response.partial_reason
            },
            "metrics": result.metrics,
            "timestamp": result.timestamp,
            "metadata": result.metadata
        }

    def _result_to_csv(self, result: EvaluationResult) -> Dict:
        return {
            "task_id": result.task_id,
            "task_type": result.task_type,
            "time_limit": result.time_limit,
            "response_status": result.response.status.value,
            "response_time": result.response.time_elapsed,
            "completion_rate": result.metrics.get("completion_rate", 0),
            "quality_score": result.metrics.get("quality_score", 0),
            "token_count": result.response.token_count,
            "token_rate": result.metrics.get("token_rate", 0),
            "output_length": len(result.response.content),
            "timestamp": result.timestamp
        }


class Evaluator:
    def __init__(self, config=None):
        self.config = config or get_config().evaluation
        self._metric_calculator = MetricCalculator()
        self._statistical_analyzer = StatisticalAnalyzer()
        self._results_manager = ResultsManager()

    def evaluate_response(
        self,
        response: LLMResponse,
        task_type: str,
        time_limit: float,
        task_id: int = 0
    ) -> EvaluationResult:
        metrics = self._metric_calculator.calculate_metrics(
            response, task_type, time_limit
        )

        result = EvaluationResult(
            task_id=task_id,
            task_type=task_type,
            time_limit=time_limit,
            response=response,
            metrics=metrics
        )

        self._results_manager.add_result(result)

        return result

    def get_analysis(self) -> Dict[str, Any]:
        results = self._results_manager.get_all_results()
        return self._statistical_analyzer.analyze_results(results)

    def get_statistics(self) -> AggregatedMetrics:
        return self._results_manager.get_statistics()

    def save_results(self, format: str = "json") -> str:
        return self._results_manager.save_results(format)

    def get_results(self) -> List[EvaluationResult]:
        return self._results_manager.get_all_results()


# Global evaluator instance
_evaluator = Evaluator()

def get_evaluator() -> Evaluator:
    return _evaluator

class EvaluatorFactory:
    @classmethod
    def create_evaluator(cls, config=None) -> Evaluator:
        return Evaluator(config)

class EvaluationController:
    def __init__(self):
        self._evaluator = get_evaluator()
        self._config = get_config().evaluation

    def start_evaluation(
        self,
        task_type: str,
        prompt: str,
        time_limit: float,
        priority: int = 1
    ) -> int:
        pass

    def get_live_metrics(self) -> Dict[str, Any]:
        stats = self._evaluator.get_statistics()
        return {
            "total_tasks": stats.total_tasks,
            "completed_tasks": stats.completed_tasks,
            "timed_out_tasks": stats.timed_out_tasks,
            "error_tasks": stats.error_tasks,
            "completion_rate": stats.avg_completion_rate,
            "avg_response_time": stats.avg_response_time,
            "avg_output_length": stats.avg_output_length
        }

    def generate_report(self) -> Dict[str, Any]:
        return {
            "statistics": self._evaluator.get_statistics(),
            "analysis": self._evaluator.get_analysis(),
            "configuration": {
                "time_limits": self._config.time_limits,
                "task_types": self._config.task_types,
                "quality_scoring_enabled": self._config.quality_scoring_enabled,
                "sample_size": self._config.sample_size
            }
        }
    
    def export_report(self, format: str = "json") -> str:
        return self._evaluator.save_results(format)
