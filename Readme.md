# Time-Constrained LLM

A research platform for studying how large language models behave under strict time constraints — measuring response quality, completion rates, and output degradation when generation is forcibly cut short.

---

## Overview

Most LLM benchmarks evaluate quality in ideal conditions. This system asks a different question: **what happens when a model runs out of time?**

By enforcing hard time limits with nanosecond precision and capturing partial outputs, this platform lets you study how response quality degrades under pressure, compare behavior across task types and difficulty levels, and analyze the tradeoff between time budgets and output completeness.

---

## Features

- **Hard time cutoffs** enforced at the nanosecond level using `time.perf_counter()`
- **Partial response capture** — records and evaluates whatever was generated before the cutoff
- **Pressure-mode prompts** — automatically injects urgency cues to test behavioral shifts
- **Side-by-side comparison** of normal vs. time-pressured outputs
- **Comprehensive metrics**: completion rate, token rate, quality score, error rate, and more
- **Statistical analysis**: confidence intervals, correlation analysis, distribution summaries
- **Dynamic task generation** across six cognitive domains with five difficulty levels
- **Google Gemini** support via the Gemini API

---

## Quick Start

**Prerequisites:** Python 3.8+, a Google API key

```bash
# 1. Clone and install
git clone https://github.com/yourusername/time-constrained-llm.git
cd time-constrained-llm
pip install -r requirements.txt

# 2. Configure your API key
echo "LLM_API_KEY=your_google_api_key" >> .env

# 3. Run a quick test
python run_pipeline.py --quick-test
```

---

## Usage

### Command Line

```bash
# Basic experiment
python run_pipeline.py \
  --name my_experiment \
  --tasks 50 \
  --time-limits 3 5 10 15 \
  --categories reasoning creative qa \
  --difficulty 2 3 4 \
  --pressure-ratio 0.5
```

### Python API

```python
from run_pipeline import get_pipeline_controller

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

print(f"Avg completion rate: {result.statistics.avg_completion_rate:.1%}")
```

### Advanced Usage

```python
from config import get_config
from core.llm import get_llm_manager
from tasks import get_task_manager

config = get_config()
config.llm.model = "gemini-1.5-flash"
config.timer.time_limits = [3, 6, 9]

task_manager = get_task_manager()
task, prompt, time_limit = task_manager.generate_task_with_enhancement(
    category="analytical",
    difficulty=3,
    time_limit=6.0,
    with_time_pressure=True
)

llm_manager = get_llm_manager()
response = llm_manager.generate_response(prompt=prompt, time_limit=time_limit)

print(f"Status:  {response.status}")          # complete / partial / timed_out
print(f"Elapsed: {response.time_elapsed:.2f}s")
print(f"Content: {response.content}")
```

---

## Example Experiments

**Time pressure effects** — compare quality across increasing time budgets:
```bash
python run_pipeline.py --name time_pressure_analysis \
  --tasks 100 --time-limits 2 5 10 20 \
  --categories reasoning creative qa --difficulty 2 3 4 --pressure-ratio 0.5
```

**Task type sensitivity** — find which domains degrade fastest under pressure:
```bash
python run_pipeline.py --name task_sensitivity \
  --tasks 150 --time-limits 3 6 9 \
  --categories reasoning creative qa analytical coding \
  --difficulty 2 3 4 5 --pressure-ratio 0.6
```

**Provider comparison** — benchmark different models side by side:
```bash
python run_pipeline.py --name provider_comparison \
  --tasks 50 --time-limits 5 \
  --categories reasoning --difficulty 3 --pressure-ratio 1.0 \
  --config '{"llm": {"provider": "google", "model": "gemini-1.5-pro"}}'
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| `completion_rate` | Fraction of expected response generated before cutoff |
| `time_to_cutoff` | Actual elapsed time at termination |
| `output_length` | Characters / tokens in the response |
| `quality_score` | Composite score (0–100) for completeness, relevance, coherence |
| `token_rate` | Tokens generated per second |
| `error_rate` | Fraction of tasks that failed entirely |
| `partial_completion` | Boolean flag for partially completed outputs |

### Accessing Analysis

```python
from core.evaluator import get_evaluator

evaluator = get_evaluator()
analysis = evaluator.get_analysis()

# Quality by time limit
for limit, metrics in analysis['time_limit_analysis'].items():
    print(f"{limit}s → quality: {metrics['avg_quality']:.1f}")

# Correlation between quality and response time
print(analysis['correlations']['quality_score-response_time'])

# Confidence intervals
ci = analysis['confidence_intervals']['quality_score']
print(f"Mean: {ci['mean']:.1f}  95% CI: [{ci['lower']:.1f}, {ci['upper']:.1f}]")
```

---

## Output Formats

Results are written to `outputs/` in both JSON and CSV.

**`outputs/responses.json`**
```json
{
  "task_id": 1,
  "task_type": "reasoning",
  "time_limit": 5.0,
  "response": {
    "content": "Partial response...",
    "status": "partial",
    "time_elapsed": 5.0,
    "token_count": 89
  },
  "metrics": {
    "completion_rate": 0.6,
    "quality_score": 55.2
  }
}
```

**`outputs/metrics.csv`**
```
task_id,task_type,time_limit,response_status,response_time,completion_rate,quality_score,token_count,token_rate
1,reasoning,5.0,partial,5.00,0.60,55.2,89,17.8
2,creative,10.0,completed,8.45,1.00,78.9,156,18.5
```

---

## Project Structure

```
time-constrained-llm/
├── run_pipeline.py        # Entry point and experiment orchestration
├── tasks.py               # Task definitions and prompt generation
├── config.py              # Configuration management
├── .env                   # API keys and environment settings
├── requirements.txt
├── core/
│   ├── llm.py             # LLM interaction, streaming, and timeouts
│   ├── timer.py           # High-precision timer and cutoff logic
│   ├── router.py          # Task routing and load distribution
│   └── evaluator.py       # Metrics collection and statistical analysis
├── prompts/
│   ├── normal.txt         # Standard prompt templates
│   └── time_pressure.txt  # Urgency-enhanced prompt templates
└── outputs/
    ├── responses.json
    ├── metrics.csv
    └── logs.txt
```

---

## Configuration

All settings can be controlled via `.env` or `config.py`.

**Key `.env` options:**

```ini
# LLM
LLM_PROVIDER=google
LLM_MODEL=gemini-1.5-flash
LLM_API_KEY=your_api_key
LLM_MAX_TOKENS=500
LLM_TEMPERATURE=0.7

# Timer
TIME_LIMITS=3,5,10,30
CUTOFF_STRATEGY=immediate
TIMER_PRECISION=0.001

# Tasks
TASK_TYPES=reasoning,creative,qa,analytical
DYNAMIC_PROMPT_GENERATION=true

# Output
ENABLE_JSON_OUTPUT=true
ENABLE_CSV_OUTPUT=true
OUTPUT_DIRECTORY=outputs
```

---

## Architecture

The system uses a multi-threaded execution model:

```
Pipeline Controller
       ↓
┌──────────────┬──────────────┬──────────────┐
│  Task Queue  │ Timer Thread │  LLM Manager │
│  & Routing   │  (per task)  │  (streaming) │
└──────────────┴──────────────┴──────────────┘
       ↓
 Evaluator → outputs/
```

Notable implementation details:

- **Thread-based timeouts** with streaming cancellation for hard cutoffs
- **Hash-based response caching** to avoid redundant API calls
- **Exponential backoff** for retries on API failures
- **Priority queue** for task scheduling with load balancing across LLM workers
- **12-factor config** with per-environment profiles (dev / test / staging / prod)

---

## Extending the System

### Add a new LLM provider

```python
from core.llm import LLMBase, LLMResponse, LLMFactory

class MyLLM(LLMBase):
    def _generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        ...

LLMFactory._instances["my_provider"] = MyLLM
```

Then set `LLM_PROVIDER=my_provider` in `.env`.

### Add a custom metric

```python
from core.evaluator import MetricCalculator

class MyMetricCalculator(MetricCalculator):
    def calculate_metrics(self, response, task_type, time_limit):
        metrics = super().calculate_metrics(response, task_type, time_limit)
        metrics['my_metric'] = self._score(response)
        return metrics
```

### Add a task category

```python
from tasks import TaskCategory, TaskDefinition
from dataclasses import dataclass

@dataclass
class MyTask(TaskDefinition):
    category: TaskCategory = TaskCategory.COMMUNICATION
    prompt_template: str = "Communicate about: {topic}"
```

---

## Troubleshooting

**API errors** — verify `LLM_API_KEY` in `.env` and check your provider's status page.

**Timeouts** — lower `LLM_TIMEOUT` or increase `TIME_LIMITS` if responses are being cut off too aggressively.

**Memory issues** — reduce `MAX_CONCURRENT_REQUESTS` or lower `LLM_MAX_TOKENS`.

**Debug mode:**
```bash
python run_pipeline.py --quick-test --log-level DEBUG
tail -f outputs/logs.txt
```

---

## Contributing

```bash
pip install -e .[dev]
pytest core/ -v --cov=core --cov-report=html
black . && flake8 .
```

---

*Built for LLM research and experimentation.*
