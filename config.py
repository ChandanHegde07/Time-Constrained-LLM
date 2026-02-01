import os
import sys
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import dotenv
class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

    @classmethod
    def from_str(cls, value: str) -> "Environment":
        try:
            return cls(value.strip().lower())
        except ValueError:
            return cls.DEVELOPMENT

class ModelProvider(Enum):
    GOOGLE = "google"

    @classmethod
    def from_str(cls, value: str) -> "ModelProvider":
        try:
            return cls(value.strip().lower())
        except ValueError:
            return cls.GOOGLE

@dataclass
class LLMConfig:
    provider: ModelProvider = ModelProvider.GOOGLE
    model: str = "gemini-1.5-flash"
    api_key: str = ""
    base_url: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class TimerConfig:
    time_limits: List[float] = field(default_factory=lambda: [3.0, 5.0, 10.0, 30.0])
    cutoff_strategy: str = "immediate"
    grace_period: float = 0.1
    precision: float = 0.01
    enable_preemptive_cutoff: bool = True


@dataclass
class TaskConfig:
    task_types: List[str] = field(default_factory=lambda: ["reasoning", "creative", "qa"])
    prompt_templates: Dict[str, str] = field(default_factory=dict)
    dynamic_prompt_generation: bool = False
    prompt_cache_size: int = 100
    enable_prompt_validation: bool = True


@dataclass
class EvaluationConfig:
    enable_metrics_collection: bool = True
    metrics: List[str] = field(default_factory=lambda: [
        "completion_rate",
        "time_to_cutoff",
        "output_length",
        "quality_score"
    ])
    quality_scoring_enabled: bool = False
    enable_statistical_analysis: bool = True
    confidence_interval: float = 0.95
    sample_size: int = 100
    benchmark_comparison_enabled: bool = False

@dataclass
class ResourceConfig:
    max_concurrent_requests: int = 5
    request_rate_limit: float = 10.0
    cpu_affinity: Optional[List[int]] = None
    memory_limit: Optional[int] = None
    enable_gpu_acceleration: bool = False
    gpu_memory_fraction: float = 0.8
    enable_memory_optimization: bool = True

@dataclass
class LoggingConfig:
    log_level: str = "INFO"
    log_file: str = "outputs/logs.txt"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_rotation: str = "daily"
    max_log_size: int = 100
    backup_count: int = 7
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    include_stack_traces: bool = False


@dataclass
class OutputConfig:
    output_directory: str = "outputs"
    response_file: str = "responses.json"
    metrics_file: str = "metrics.csv"
    enable_json_output: bool = True
    enable_csv_output: bool = True
    compression_level: int = 0
    backup_strategy: str = "timestamp"
    max_storage_size: int = 1000
    cleanup_age_days: int = 30

@dataclass
class FeatureFlags:
    enable_early_cutoff: bool = True
    enable_partial_response_saving: bool = True
    enable_prompt_enhancement: bool = False
    enable_caching: bool = True
    enable_rate_limiting: bool = True
    enable_throttling: bool = True
    enable_error_handling: bool = True
    enable_retry_logic: bool = True
    enable_performance_tracking: bool = True


@dataclass
class SecurityConfig:
    api_key_rotation_enabled: bool = False
    api_key_rotation_interval: int = 30
    data_encryption_enabled: bool = False
    encryption_algorithm: str = "AES-256"
    audit_logging_enabled: bool = False
    audit_log_path: str = "audit.log"
    data_retention_period: int = 90
    enable_input_validation: bool = True
    sanitize_output: bool = False


@dataclass
class SystemConfig:
    environment: Environment = Environment.DEVELOPMENT
    llm: LLMConfig = field(default_factory=LLMConfig)
    timer: TimerConfig = field(default_factory=TimerConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    version: str = "1.0.0"


class ConfigManager:
    _instance: Optional["ConfigManager"] = None
    _config: Optional[SystemConfig] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._config = None
        return cls._instance

    def __init__(self):
        if ConfigManager._config is None:
            self._load_config()

    def _load_config(self) -> None:
        # Load .env file
        dotenv.load_dotenv()

        # Determine environment
        env_str = os.getenv("ENVIRONMENT", "development")
        environment = Environment.from_str(env_str)

        # Create base config
        config = SystemConfig(environment=environment)

        # Load LLM configuration
        config.llm.provider = ModelProvider.from_str(
            os.getenv("LLM_PROVIDER", "google")
        )
        config.llm.model = os.getenv("LLM_MODEL", "gemini-1.5-flash")
        config.llm.api_key = os.getenv("LLM_API_KEY", config.llm.api_key)
        config.llm.base_url = os.getenv("LLM_BASE_URL")
        config.llm.max_tokens = int(os.getenv("LLM_MAX_TOKENS", config.llm.max_tokens))
        config.llm.temperature = float(os.getenv("LLM_TEMPERATURE", config.llm.temperature))
        config.llm.top_p = float(os.getenv("LLM_TOP_P", config.llm.top_p))
        config.llm.frequency_penalty = float(os.getenv("LLM_FREQUENCY_PENALTY", config.llm.frequency_penalty))
        config.llm.presence_penalty = float(os.getenv("LLM_PRESENCE_PENALTY", config.llm.presence_penalty))
        config.llm.timeout = float(os.getenv("LLM_TIMEOUT", config.llm.timeout))
        config.llm.max_retries = int(os.getenv("LLM_MAX_RETRIES", config.llm.max_retries))
        config.llm.retry_delay = float(os.getenv("LLM_RETRY_DELAY", config.llm.retry_delay))

        # Load timer configuration
        config.timer.time_limits = self._parse_list(
            os.getenv("TIME_LIMITS"), [3.0, 5.0, 10.0, 30.0], float
        )
        config.timer.cutoff_strategy = os.getenv("CUTOFF_STRATEGY", config.timer.cutoff_strategy)
        config.timer.grace_period = float(os.getenv("GRACE_PERIOD", config.timer.grace_period))
        config.timer.precision = float(os.getenv("TIMER_PRECISION", config.timer.precision))
        config.timer.enable_preemptive_cutoff = self._parse_bool(
            os.getenv("ENABLE_PREEMPTIVE_CUTOFF"), config.timer.enable_preemptive_cutoff
        )

        # Load task configuration
        config.task.task_types = self._parse_list(
            os.getenv("TASK_TYPES"), ["reasoning", "creative", "qa"], str
        )
        config.task.dynamic_prompt_generation = self._parse_bool(
            os.getenv("DYNAMIC_PROMPT_GENERATION"), config.task.dynamic_prompt_generation
        )
        config.task.prompt_cache_size = int(os.getenv("PROMPT_CACHE_SIZE", config.task.prompt_cache_size))
        config.task.enable_prompt_validation = self._parse_bool(
            os.getenv("ENABLE_PROMPT_VALIDATION"), config.task.enable_prompt_validation
        )

        # Load evaluation configuration
        config.evaluation.enable_metrics_collection = self._parse_bool(
            os.getenv("ENABLE_METRICS_COLLECTION"), config.evaluation.enable_metrics_collection
        )
        config.evaluation.metrics = self._parse_list(
            os.getenv("METRICS"), [
                "completion_rate",
                "time_to_cutoff",
                "output_length",
                "quality_score"
            ], str
        )
        config.evaluation.quality_scoring_enabled = self._parse_bool(
            os.getenv("QUALITY_SCORING_ENABLED"), config.evaluation.quality_scoring_enabled
        )
        config.evaluation.enable_statistical_analysis = self._parse_bool(
            os.getenv("ENABLE_STATISTICAL_ANALYSIS"), config.evaluation.enable_statistical_analysis
        )
        config.evaluation.confidence_interval = float(os.getenv("CONFIDENCE_INTERVAL", config.evaluation.confidence_interval))
        config.evaluation.sample_size = int(os.getenv("SAMPLE_SIZE", config.evaluation.sample_size))
        config.evaluation.benchmark_comparison_enabled = self._parse_bool(
            os.getenv("BENCHMARK_COMPARISON_ENABLED"), config.evaluation.benchmark_comparison_enabled
        )

        # Load resource configuration
        config.resources.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", config.resources.max_concurrent_requests))
        config.resources.request_rate_limit = float(os.getenv("REQUEST_RATE_LIMIT", config.resources.request_rate_limit))
        config.resources.cpu_affinity = self._parse_list(
            os.getenv("CPU_AFFINITY"), None, int
        )
        config.resources.memory_limit = self._parse_int(os.getenv("MEMORY_LIMIT"))
        config.resources.enable_gpu_acceleration = self._parse_bool(
            os.getenv("ENABLE_GPU_ACCELERATION"), config.resources.enable_gpu_acceleration
        )
        config.resources.gpu_memory_fraction = float(os.getenv("GPU_MEMORY_FRACTION", config.resources.gpu_memory_fraction))
        config.resources.enable_memory_optimization = self._parse_bool(
            os.getenv("ENABLE_MEMORY_OPTIMIZATION"), config.resources.enable_memory_optimization
        )

        # Load logging configuration
        config.logging.log_level = os.getenv("LOG_LEVEL", config.logging.log_level)
        config.logging.log_file = os.getenv("LOG_FILE", config.logging.log_file)
        config.logging.log_format = os.getenv("LOG_FORMAT", config.logging.log_format)
        config.logging.log_rotation = os.getenv("LOG_ROTATION", config.logging.log_rotation)
        config.logging.max_log_size = int(os.getenv("MAX_LOG_SIZE", config.logging.max_log_size))
        config.logging.backup_count = int(os.getenv("BACKUP_COUNT", config.logging.backup_count))
        config.logging.enable_file_logging = self._parse_bool(
            os.getenv("ENABLE_FILE_LOGGING"), config.logging.enable_file_logging
        )
        config.logging.enable_console_logging = self._parse_bool(
            os.getenv("ENABLE_CONSOLE_LOGGING"), config.logging.enable_console_logging
        )
        config.logging.include_stack_traces = self._parse_bool(
            os.getenv("INCLUDE_STACK_TRACES"), config.logging.include_stack_traces
        )

        # Load output configuration
        config.output.output_directory = os.getenv("OUTPUT_DIRECTORY", config.output.output_directory)
        config.output.response_file = os.getenv("RESPONSE_FILE", config.output.response_file)
        config.output.metrics_file = os.getenv("METRICS_FILE", config.output.metrics_file)
        config.output.enable_json_output = self._parse_bool(
            os.getenv("ENABLE_JSON_OUTPUT"), config.output.enable_json_output
        )
        config.output.enable_csv_output = self._parse_bool(
            os.getenv("ENABLE_CSV_OUTPUT"), config.output.enable_csv_output
        )
        config.output.compression_level = int(os.getenv("COMPRESSION_LEVEL", config.output.compression_level))
        config.output.backup_strategy = os.getenv("BACKUP_STRATEGY", config.output.backup_strategy)
        config.output.max_storage_size = int(os.getenv("MAX_STORAGE_SIZE", config.output.max_storage_size))
        config.output.cleanup_age_days = int(os.getenv("CLEANUP_AGE_DAYS", config.output.cleanup_age_days))

        # Load feature flags
        config.features.enable_early_cutoff = self._parse_bool(
            os.getenv("ENABLE_EARLY_CUTOFF"), config.features.enable_early_cutoff
        )
        config.features.enable_partial_response_saving = self._parse_bool(
            os.getenv("ENABLE_PARTIAL_RESPONSE_SAVING"), config.features.enable_partial_response_saving
        )
        config.features.enable_prompt_enhancement = self._parse_bool(
            os.getenv("ENABLE_PROMPT_ENHANCEMENT"), config.features.enable_prompt_enhancement
        )
        config.features.enable_caching = self._parse_bool(
            os.getenv("ENABLE_CACHING"), config.features.enable_caching
        )
        config.features.enable_rate_limiting = self._parse_bool(
            os.getenv("ENABLE_RATE_LIMITING"), config.features.enable_rate_limiting
        )
        config.features.enable_throttling = self._parse_bool(
            os.getenv("ENABLE_THROTTLING"), config.features.enable_throttling
        )
        config.features.enable_error_handling = self._parse_bool(
            os.getenv("ENABLE_ERROR_HANDLING"), config.features.enable_error_handling
        )
        config.features.enable_retry_logic = self._parse_bool(
            os.getenv("ENABLE_RETRY_LOGIC"), config.features.enable_retry_logic
        )
        config.features.enable_performance_tracking = self._parse_bool(
            os.getenv("ENABLE_PERFORMANCE_TRACKING"), config.features.enable_performance_tracking
        )

        # Load security configuration
        config.security.api_key_rotation_enabled = self._parse_bool(
            os.getenv("API_KEY_ROTATION_ENABLED"), config.security.api_key_rotation_enabled
        )
        config.security.api_key_rotation_interval = int(os.getenv("API_KEY_ROTATION_INTERVAL", config.security.api_key_rotation_interval))
        config.security.data_encryption_enabled = self._parse_bool(
            os.getenv("DATA_ENCRYPTION_ENABLED"), config.security.data_encryption_enabled
        )
        config.security.encryption_algorithm = os.getenv("ENCRYPTION_ALGORITHM", config.security.encryption_algorithm)
        config.security.audit_logging_enabled = self._parse_bool(
            os.getenv("AUDIT_LOGGING_ENABLED"), config.security.audit_logging_enabled
        )
        config.security.audit_log_path = os.getenv("AUDIT_LOG_PATH", config.security.audit_log_path)
        config.security.data_retention_period = int(os.getenv("DATA_RETENTION_PERIOD", config.security.data_retention_period))
        config.security.enable_input_validation = self._parse_bool(
            os.getenv("ENABLE_INPUT_VALIDATION"), config.security.enable_input_validation
        )
        config.security.sanitize_output = self._parse_bool(
            os.getenv("SANITIZE_OUTPUT"), config.security.sanitize_output
        )

        # Apply environment-specific overrides
        self._apply_environment_overrides(config)

        # Validate configuration
        self._validate_config(config)

        ConfigManager._config = config

    def _parse_list(
        self,
        value: Optional[str],
        default: Optional[Union[List[Any], None]],
        type_converter: type
    ) -> Optional[Union[List[Any], None]]:
        if value is None:
            return default
        try:
            return [type_converter(x.strip()) for x in value.split(",") if x.strip()]
        except Exception as e:
            print(f"Warning: Failed to parse list value '{value}': {e}", file=sys.stderr)
            return default

    def _parse_bool(self, value: Optional[str], default: bool) -> bool:
        if value is None:
            return default
        value = value.strip().lower()
        true_values = ["true", "1", "yes", "on"]
        false_values = ["false", "0", "no", "off"]
        if value in true_values:
            return True
        if value in false_values:
            return False
        print(f"Warning: Unknown boolean value '{value}', using default: {default}", file=sys.stderr)
        return default

    def _parse_int(self, value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _apply_environment_overrides(self, config: SystemConfig) -> None:
        """Apply environment-specific configuration overrides."""
        if config.environment == Environment.PRODUCTION:
            config.resources.max_concurrent_requests = 10
            config.resources.request_rate_limit = 20.0
            config.logging.log_level = "WARNING"
            config.logging.include_stack_traces = False
            config.llm.timeout = 30.0
            config.llm.max_retries = 5
            config.timer.precision = 0.005
        elif config.environment == Environment.TESTING:
            config.resources.max_concurrent_requests = 2
            config.resources.request_rate_limit = 5.0
            config.logging.log_level = "DEBUG"
            config.llm.timeout = 10.0
            config.llm.max_retries = 1
        elif config.environment == Environment.STAGING:
            config.resources.max_concurrent_requests = 5
            config.resources.request_rate_limit = 15.0
            config.logging.log_level = "INFO"
            config.llm.timeout = 45.0

    def _validate_config(self, config: SystemConfig) -> None:
        errors = []

        # Validate time limits
        for limit in config.timer.time_limits:
            if limit <= 0:
                errors.append(f"Time limit must be positive: {limit}")
        if len(config.timer.time_limits) == 0:
            errors.append("At least one time limit must be specified")

        # Validate LLM configuration
        if not config.llm.api_key:
            errors.append("API key is required for Google LLM provider")
        if config.llm.max_tokens <= 0:
            errors.append(f"Max tokens must be positive: {config.llm.max_tokens}")
        if not (0 <= config.llm.temperature <= 2):
            errors.append(f"Temperature must be between 0 and 2: {config.llm.temperature}")
        if not (0 <= config.llm.top_p <= 1):
            errors.append(f"Top-p must be between 0 and 1: {config.llm.top_p}")
        if config.llm.timeout <= 0:
            errors.append(f"Timeout must be positive: {config.llm.timeout}")
        if config.llm.max_retries < 0:
            errors.append(f"Max retries cannot be negative: {config.llm.max_retries}")

        # Validate resource configuration
        if config.resources.max_concurrent_requests <= 0:
            errors.append(f"Max concurrent requests must be positive: {config.resources.max_concurrent_requests}")
        if config.resources.request_rate_limit <= 0:
            errors.append(f"Request rate limit must be positive: {config.resources.request_rate_limit}")

        # Validate logging configuration
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.logging.log_level.upper() not in valid_log_levels:
            errors.append(f"Invalid log level: {config.logging.log_level}")

        # Validate output configuration
        if config.output.compression_level < 0 or config.output.compression_level > 9:
            errors.append(f"Compression level must be between 0 and 9: {config.output.compression_level}")
        if config.output.cleanup_age_days <= 0:
            errors.append(f"Cleanup age must be positive: {config.output.cleanup_age_days}")

        # Validate security configuration
        if config.security.data_encryption_enabled and not config.security.encryption_algorithm:
            errors.append("Encryption algorithm required when data encryption is enabled")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            if config.environment in [Environment.PRODUCTION, Environment.STAGING]:
                raise RuntimeError(error_msg)
            else:
                print(f"Warning: {error_msg}", file=sys.stderr)

    @property
    def config(self) -> SystemConfig:
        if ConfigManager._config is None:
            self._load_config()
        return ConfigManager._config

    def reload(self) -> None:
        ConfigManager._config = None
        self._load_config()

    def get(self, path: str, default: Any = None) -> Any:
        parts = path.split(".")
        value = self.config
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return default
        return value

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        config_obj = self.config
        for part in parts[:-1]:
            if hasattr(config_obj, part):
                config_obj = getattr(config_obj, part)
            else:
                raise AttributeError(f"Configuration path '{'.'.join(parts[:-1])}' not found")
        setattr(config_obj, parts[-1], value)


# Global singleton instance
_config_manager = ConfigManager()

# Module-level convenience access
config = _config_manager.config

def get_config() -> SystemConfig:
    return config

def reload_config() -> None:
    _config_manager.reload()

def get_config_value(path: str, default: Any = None) -> Any:
    return _config_manager.get(path, default)

def set_config_value(path: str, value: Any) -> None:
    _config_manager.set(path, value)

# Create directories for outputs and logs on import
def _ensure_directories() -> None:
    try:
        output_dir = Path(config.output.output_directory)
        output_dir.mkdir(exist_ok=True)
        # Check if we need to create separate log directory
        log_path = Path(config.logging.log_file)
        log_path.parent.mkdir(exist_ok=True)
    except Exception as e:
        print(f"Warning: Failed to create directories: {e}", file=sys.stderr)

# Initialize directories when module is imported
_ensure_directories()
