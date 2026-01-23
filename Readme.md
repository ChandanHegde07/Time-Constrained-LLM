# ⏱Time-Constrained LLM System

A lightweight LLM pressure-testing system that simulates strict time constraints to study how large language models behave under urgency and forced cutoffs.

Built as a minimal prototype, this project helps analyze response degradation, partial outputs, and decision quality when an LLM is rushed.

---

## What Does It Do?

This system enforces hard time limits on LLM responses and captures how outputs change under pressure.

It allows you to:

- **Simulate real-time deadlines** during generation
- **Force early cutoffs** and record partial responses
- **Compare** normal vs time-pressured prompts
- **Measure** completion, timing, and output length

---

## How It Works

1. A task prompt is selected (reasoning, creative, or QA)
2. A countdown timer starts
3. The LLM is prompted to generate a response
4. If time expires, generation is stopped immediately
5. Partial or complete outputs are logged and evaluated

---

## Core Components

| Component | Purpose |
|-----------|---------|
| **LLM Module** | Handles model interaction and response generation |
| **Timer Module** | Enforces time limits and cutoffs |
| **Task Module** | Stores predefined test prompts |
| **Evaluator Module** | Tracks completion status and timing metrics |

---

## Use Cases

- Studying LLM behavior under stress
- Comparing model robustness under time pressure
- Demonstrating partial-output handling
- Educational demos and research experiments

---

## Why This Matters

Real-world AI systems often operate under latency and deadline constraints. This project provides a simple framework to explore how those constraints impact LLM performance and reliability.

---

## Getting Started
```bash
# Clone the repository
git clone https://github.com/yourusername/time-constrained-llm.git

# Navigate to the project directory
cd time-constrained-llm

# Install dependencies
pip install -r requirements.txt

# Run the system
python main.py
```

---

## Example Output
```
Task: Reasoning Problem
Time Limit: 5 seconds
Status: Partial completion (60%)
Response Length: 234 characters
Time Elapsed: 5.00s
```

---

## Configuration

Customize test parameters in `config.yaml`:
```yaml
time_limits: [3, 5, 10, 30]  # seconds
task_types: [reasoning, creative, qa]
model: gpt-4
max_tokens: 500
```

---

## Metrics Tracked

- **Completion Rate**: Percentage of response completed
- **Time to Cutoff**: Actual time elapsed
- **Output Length**: Character/token count
- **Quality Score**: Optional manual evaluation

---

**Built with ⚡ for LLM research and experimentation**