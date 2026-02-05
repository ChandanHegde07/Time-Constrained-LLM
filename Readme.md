# Time-Constrained LLM System

A comprehensive LLM pressure-testing system that simulates strict time constraints to study how large language models behave under urgency and forced cutoffs.

Built as a production-grade platform, this project helps analyze response degradation, partial outputs, and decision quality when an LLM is rushed, with comprehensive metrics collection and statistical analysis.

---

## What Does It Do?

This system enforces hard time limits on LLM responses and captures how outputs change under pressure with precision timing and detailed metrics.

Key capabilities:

- **Simulate real-time deadlines** with nanosecond-level accuracy
- **Force early cutoffs** and record partial responses with reasoning
- **Compare normal vs time-pressured prompts** automatically
- **Measure comprehensive metrics**: completion rate, timing, output quality, token efficiency
- **Support Google LLM provider**: Google Gemini models for time-constrained testing
- **Advanced statistical analysis**: correlation, distribution, confidence intervals
- **Dynamic task generation**: diverse cognitive domains, difficulty levels, response formats

---

## How It Works

1. **Task Selection**: Choose from predefined task categories or generate dynamically
2. **Timer Initialization**: High-precision timer starts with configurable grace period
3. **Response Generation**: LLM processes prompt with streaming and timeout monitoring
4. **Cutoff Enforcement**: Generation stops immediately if time limit exceeded
5. **Partial Response Handling**: Record and evaluate content up to cutoff point
6. **Metrics Collection**: Comprehensive performance data captured in real-time
7. **Analysis & Reporting**: Statistical analysis and visualization of results

---

## Core Components

### LLM Module ([`core/llm.py`](core/llm.py))
- **Provider Support**: Google (Gemini models)
- **Time Management**: Thread-based timeout with streaming cancellation
- **Response Types**: Complete, partial, timed-out, and error responses
- **Caching**: Response caching with cache key management
- **Retry Logic**: Configurable retry with exponential backoff

### Timer Module ([`core/timer.py`](core/timer.py))
- **Precision Timing**: Nanosecond-level accuracy using `time.perf_counter()`
- **Hard Cutoffs**: Preemptive timeout with configurable grace period
- **State Management**: IDLE → RUNNING → COMPLETED/TIMED_OUT/PAUSED states
- **Callbacks**: Time-based callbacks for monitoring and control
- **Context Manager**: Pythonic timer usage with `with` statement
- **Decorator**: Easy integration with existing functions

### Router Module ([`core/router.py`](core/router.py))
- **Routing Strategies**: Round-robin, priority, load-balance, performance-based
- **Task Queue**: Priority queue with dynamic scheduling
- **Load Management**: Connection limits and rate control
- **Distribution**: Even workload distribution across LLM instances
- **Orchestration**: Multi-worker execution with resource management

### Evaluator Module ([`core/evaluator.py`](core/evaluator.py))
- **Metrics Collection**:
  - Completion rate
  - Time to cutoff
  - Output length (tokens/characters)
  - Quality scoring (custom metrics)
  - Token generation rate
  - Error rate

- **Statistical Analysis**:
  - Descriptive statistics
  - Time limit effects
  - Task type performance
  - Correlation analysis
  - Confidence intervals
  - Distribution analysis

- **Quality Scoring**:
  - Completeness assessment
  - Relevance scoring
  - Coherence evaluation
  - Accuracy estimation

### Task Manager ([`tasks.py`](tasks.py))
- **Task Library**: Predefined tasks across cognitive domains
- **Dynamic Generation**: Context-aware task creation
- **Prompt Enhancement**: Time pressure cues and urgency triggers
- **Categories**: Reasoning, creative, QA, analytical, coding, problem-solving
- **Difficulty Levels**: 1-5 scale with adaptive generation
- **Static Prompts**: File-based prompt management

### Configuration ([`config.py`](config.py) & [`.env`](.env))
- **Environment-aware**: Development, testing, staging, production profiles
- **Type-safe**: Dataclass-based configuration with validation
- **Feature Flags**: Dynamic feature toggles
- **Resource Limits**: Connection limits, memory constraints, GPU configuration
- **Security**: API key management, data encryption, audit logging

---

## Project Structure

```
time-constrained-llm/
├── config.py              # Configuration management
├── run_pipeline.py        # Main pipeline orchestration
├── tasks.py              # Task definitions and generation
├── .env                  # Environment variables
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
├── Readme.md            # Documentation (this file)
├── core/                # Core modules
│   ├── llm.py          # LLM interaction
│   ├── timer.py        # Timing and cutoff management
│   ├── router.py       # Task routing and distribution
│   └── evaluator.py    # Metrics and analysis
├── prompts/            # Prompt templates
│   ├── normal.txt      # Normal task prompts
│   └── time_pressure.txt # Time pressure enhanced prompts
└── outputs/            # Results and logs
    ├── responses.json  # Generated responses
    ├── metrics.csv     # Performance metrics
    └── logs.txt        # System logs
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Google API key (for Google Gemini integration)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/time-constrained-llm.git
   cd time-constrained-llm
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API keys in `.env` file:
   ```
   LLM_API_KEY=your_google_api_key
   LLM_PROVIDER=google
   LLM_MODEL=gemini-1.5-flash
   ```

### Quick Start

Run a quick test with default settings:
```bash
python run_pipeline.py --quick-test
```

Run a comprehensive test:
```bash
python run_pipeline.py --name my_experiment --tasks 50 --time-limits 3 5 10 15 --categories reasoning creative qa --difficulty 2 3 4 --pressure-ratio 0.5
```

### Using Custom Configuration

Create a custom experiment configuration:
```python
from run_pipeline import ExperimentFactory

# Create and run a quick test
experiment = ExperimentFactory.create_quick_test("quick_test_001")
result = experiment.run()
print(f"Experiment completed in {result.duration:.2f} seconds")
print(f"Total tasks: {result.statistics.total_tasks}")
print(f"Completed tasks: {result.statistics.completed_tasks}")
```

---

## Usage Examples

### Basic Usage

```python
from run_pipeline import get_pipeline_controller

# Create and run a simple experiment
controller = get_pipeline_controller()
result = controller.run_experiment(
    name="basic_experiment",
    config={
        "task_count": 10,
        "time_limits": [5, 10],
        "task_categories": ["reasoning", "qa"],
        "difficulty_levels": [2, 3],
        "time_pressure_ratio": 0.5
    }
)

print(f"Experiment Results: {result.statistics}")
```

### Advanced Configuration

```python
from config import get_config
from tasks import generate_task
from core.llm import get_llm_manager

# Custom configuration
config = get_config()
config.llm.model = "gemini-1.5-flash"
config.timer.time_limits = [3, 6, 9]
config.timer.precision = 0.001

# Generate and execute a task
task_manager = get_task_manager()
task, prompt, time_limit = task_manager.generate_task_with_enhancement(
    category="analytical",
    difficulty=3,
    time_limit=6.0,
    with_time_pressure=True
)

# Execute with LLM manager
llm_manager = get_llm_manager()
response = llm_manager.generate_response(
    prompt=prompt,
    time_limit=time_limit,
    use_cache=True
)

print(f"Response Status: {response.status}")
print(f"Content: {response.content}")
print(f"Time Elapsed: {response.time_elapsed:.2f}s")
```
---

## Configuration Options

### Environment Variables

Key configuration options in `.env` file:

```
# Environment
ENVIRONMENT=development

# LLM Configuration
LLM_PROVIDER=google
LLM_MODEL=gemini-1.5-flash
LLM_API_KEY=your_api_key
LLM_MAX_TOKENS=500
LLM_TEMPERATURE=0.7
LLM_TIMEOUT=60

# Timer Configuration
TIME_LIMITS=3,5,10,30
CUTOFF_STRATEGY=immediate
TIMER_PRECISION=0.001

# Task Configuration
TASK_TYPES=reasoning,creative,qa,analytical
DYNAMIC_PROMPT_GENERATION=true

# Resource Configuration
MAX_CONCURRENT_REQUESTS=5
REQUEST_RATE_LIMIT=10

# Output Configuration
ENABLE_JSON_OUTPUT=true
ENABLE_CSV_OUTPUT=true
OUTPUT_DIRECTORY=outputs
```

### Advanced Configuration in config.py

```python
from config import get_config, reload_config

# Access configuration
config = get_config()
print(f"Current provider: {config.llm.provider}")
print(f"Model: {config.llm.model}")

# Modify configuration
config.llm.temperature = 0.5
config.timer.time_limits = [2, 4, 6, 8]

# Reload from environment
reload_config()
```

---

## Metrics and Analysis

### Metrics Tracked

| Metric | Description |
|--------|-------------|
| **Completion Rate** | Percentage of response completed before cutoff |
| **Time to Cutoff** | Actual generation time until termination |
| **Output Length** | Number of characters/tokens generated |
| **Quality Score** | Subjective quality rating (0-100) |
| **Token Rate** | Generation speed in tokens per second |
| **Error Rate** | Percentage of failed responses |
| **Partial Completion** | Indicator of partial output |

### Analysis Types

```python
from core.evaluator import get_evaluator

evaluator = get_evaluator()
results = evaluator.get_results()

# Calculate statistics
stats = evaluator.get_statistics()
print(f"Total tasks: {stats.total_tasks}")
print(f"Avg completion rate: {stats.avg_completion_rate:.1%}")

# Get detailed analysis
analysis = evaluator.get_analysis()

# Time limit effects
time_analysis = analysis['time_limit_analysis']
for time_limit, metrics in time_analysis.items():
    print(f"{time_limit}s - Quality: {metrics['avg_quality']:.1f}")

# Correlation analysis
correlations = analysis['correlations']
print(f"Quality-Time correlation: {correlations['quality_score-response_time']}")

# Confidence intervals
confidence = analysis['confidence_intervals']
print(f"Mean quality: {confidence['quality_score']['mean']:.1f}")
print(f"CI: [{confidence['quality_score']['lower']:.1f}, {confidence['quality_score']['upper']:.1f}]")
```

---

## Output Formats

### JSON Output
```json
[
  {
    "task_id": 1,
    "task_type": "reasoning",
    "time_limit": 5.0,
    "response": {
      "content": "Partial response...",
      "status": "partial",
      "time_elapsed": 5.0,
      "token_count": 89,
      "model": "gpt-4"
    },
    "metrics": {
      "completion_rate": 0.6,
      "quality_score": 55.2
    }
  }
]
```

### CSV Output
```csv
task_id,task_type,time_limit,response_status,response_time,completion_rate,quality_score,token_count,token_rate,output_length
1,reasoning,5.0,partial,5.00,0.60,55.2,89,17.8,234
2,creative,10.0,completed,8.45,1.00,78.9,156,18.5,412
```

---

## Research Applications

### Example Experiments

#### Experiment 1: Time Pressure Effects
```bash
python run_pipeline.py --name time_pressure_analysis --tasks 100 --time-limits 2 5 10 20 --categories reasoning creative qa --difficulty 2 3 4 --pressure-ratio 0.5
```

#### Experiment 2: Provider Comparison
```bash
python run_pipeline.py --name provider_comparison --tasks 50 --time-limits 5 --categories reasoning --difficulty 3 --pressure-ratio 1.0 --config "{\"llm\": {\"provider\": \"anthropic\", \"model\": \"claude-3-opus-20240229\"}}"
```

#### Experiment 3: Task Type Sensitivity
```bash
python run_pipeline.py --name task_sensitivity --tasks 150 --time-limits 3 6 9 --categories reasoning creative qa analytical coding --difficulty 2 3 4 5 --pressure-ratio 0.6
```

---

## Architecture Details

### Threading Architecture

The system uses a multi-threaded approach for LLM interaction with timeouts:

```
┌─────────────────────────────────────────┐
│          Main Execution Thread          │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│          Pipeline Controller            │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────┬─────────────────┬─────────────────┐
│  Task Queue     │  Timer Thread   │  LLM Manager    │
│  Management     │  (Per Task)     │  (Generation)   │
└─────────────────┴─────────────────┴─────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│         LLM Response Generation         │
└─────────────────────────────────────────┘
```

### Configuration Management

The system follows 12-factor app principles:

- **Environment-aware**: Different profiles for dev, test, staging, production
- **Validation**: Configuration validation on startup
- **Hot Reload**: Dynamic configuration changes without restart
- **Type Safety**: Dataclass-based configuration with type checks

### Performance Optimization

1. **Response Caching**: Hash-based caching using prompt + params
2. **Resource Pooling**: Connection pooling and rate limiting
3. **Streaming**: Real-time response handling with cancellation
4. **Retry Logic**: Exponential backoff for failed API calls
5. **Memory Management**: Efficient resource allocation and cleanup

---

## Customization

### Adding New LLM Provider

```python
from core.llm import LLMBase, LLMResponse, LLMResponseStatus

class CustomLLM(LLMBase):
    def _generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        # Implementation for custom provider
        pass

# Register with factory
from core.llm import LLMFactory
LLMFactory._instances["custom"] = CustomLLM

# Configure in .env
LLM_PROVIDER=custom
LLM_MODEL=custom-model
```

### Adding New Metrics

```python
from core.evaluator import MetricCalculator

class CustomMetricCalculator(MetricCalculator):
    def _calculate_custom_metric(self, response, task_type) -> float:
        # Metric calculation logic
        return 0.0

    def calculate_metrics(self, response, task_type, time_limit):
        metrics = super().calculate_metrics(response, task_type, time_limit)
        metrics['custom_metric'] = self._calculate_custom_metric(response, task_type)
        return metrics
```

### Creating New Task Categories

```python
from tasks import TaskCategory, TaskDefinition

@dataclass
class CustomTaskDefinition(TaskDefinition):
    task_id: str
    category: TaskCategory = TaskCategory.COMMUNICATION
    name: str = "Communication Task"
    description: str = "Communication-related task"
    prompt_template: str = "Communicate about: {topic}"
```

---

## Troubleshooting

### Common Issues

**API Connection Errors**:
- Check API key in `.env` file
- Verify internet connectivity
- Check API provider status page

**Timeout Issues**:
- Reduce `LLM_TIMEOUT` value
- Increase `TIME_LIMITS` in config
- Check network latency

**Memory Issues**:
- Reduce `MAX_CONCURRENT_REQUESTS`
- Limit `LLM_MAX_TOKENS`
- Enable `ENABLE_MEMORY_OPTIMIZATION=true`

### Debugging

```bash
python run_pipeline.py --quick-test --log-level DEBUG
```

**Checking Logs**:
```bash
tail -f outputs/logs.txt
```

---

## Contributing

### Development Setup

```bash
# Install dev dependencies
pip install -e .[dev]

# Run tests
pytest core/ -v

# Run with coverage
pytest core/ -v --cov=core --cov-report=html

# Format code
black .
flake8 .
```

### Architecture Principles

1. **Separation of Concerns**: Each module handles specific responsibility
2. **Testability**: Dependency injection and abstract interfaces
3. **Extensibility**: Factory pattern for adding new features
4. **Performance**: Thread-safe operations, efficient resource usage
5. **Maintainability**: Clear architecture, comprehensive documentation

---

## Future Enhancements

### Planned Features

1. **Model Comparison Dashboard**: Visual comparison of different LLMs
2. **Real-time Monitoring**: Web-based interface for experiment tracking
3. **Advanced Analysis**: Machine learning-based quality scoring
4. **Custom Metrics**: Plugin system for domain-specific metrics
5. **Resource Optimization**: GPU utilization and memory management
6. **Distributed Execution**: Multi-node processing for large experiments

### Research Roadmap

- Response quality vs. time pressure tradeoff
- Partial response comprehensiveness
- Generation speed across difficulty levels
- Cognitive load impact on LLM performance

---

**Built for LLM research and experimentation**
