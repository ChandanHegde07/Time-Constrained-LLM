import json
import random
import string
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from config import get_config


class TaskCategory(Enum):
    REASONING = "reasoning"
    CREATIVE = "creative"
    QA = "qa"
    ANALYTICAL = "analytical"
    PROBLEM_SOLVING = "problem_solving"
    CODING = "coding"
    SUMMARIZATION = "summarization"
    DECISION_MAKING = "decision_making"
    COMPREHENSION = "comprehension"
    COMMUNICATION = "communication"


@dataclass
class TaskDefinition:
    task_id: str
    category: TaskCategory
    name: str
    description: str
    prompt_template: str
    difficulty: int  # 1-5 scale
    expected_complexity: str  # Low, Medium, High
    time_estimation: float  # Expected completion time in seconds
    sample_responses: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskGenerator:
    def __init__(self, config=None):
        self.config = config or get_config().task
        self._task_templates = self._load_task_templates()

    def _load_task_templates(self) -> List[TaskDefinition]:
        """Load predefined task templates from configuration or files."""
        return [
            TaskDefinition(
                task_id="reasoning_001",
                category=TaskCategory.REASONING,
                name="Logical Reasoning",
                description="Solve a logical reasoning problem",
                prompt_template="Solve the following logical reasoning problem:\n\n{problem}",
                difficulty=3,
                expected_complexity="Medium",
                time_estimation=60,
                sample_responses=[
                    "The answer is X because...",
                    "Based on logical analysis...",
                    "Therefore, the conclusion is..."
                ],
                metadata={"domain": "logic", "subdomain": "deduction"}
            ),
            TaskDefinition(
                task_id="reasoning_002",
                category=TaskCategory.REASONING,
                name="Rapid Reasoning",
                description="Quick logical deduction under pressure",
                prompt_template="Solve this quickly! You have {time_limit} seconds:\n\n{problem}",
                difficulty=2,
                expected_complexity="Low",
                time_estimation=5,
                sample_responses=[
                    "Answer: X",
                    "Quick deduction: ...",
                    "Rapid conclusion: ..."
                ],
                metadata={"domain": "logic", "subdomain": "rapid_deduction"}
            ),

            TaskDefinition(
                task_id="creative_001",
                category=TaskCategory.CREATIVE,
                name="Creative Writing",
                description="Generate creative text based on a prompt",
                prompt_template="Write a creative story or poem about:\n\n{topic}",
                difficulty=4,
                expected_complexity="High",
                time_estimation=120,
                sample_responses=[
                    "Once upon a time...",
                    "In a faraway land...",
                    "The story begins..."
                ],
                metadata={"domain": "creative", "subdomain": "storytelling"}
            ),

            TaskDefinition(
                task_id="qa_001",
                category=TaskCategory.QA,
                name="General Knowledge",
                description="Answer general knowledge questions",
                prompt_template="Answer the following question:\n\n{question}",
                difficulty=2,
                expected_complexity="Low",
                time_estimation=30,
                sample_responses=[
                    "The answer is...",
                    "According to my knowledge...",
                    "From available information..."
                ],
                metadata={"domain": "knowledge", "subdomain": "general"}
            ),

            TaskDefinition(
                task_id="analytical_001",
                category=TaskCategory.ANALYTICAL,
                name="Data Analysis",
                description="Analyze and interpret data",
                prompt_template="Analyze the following data and answer questions:\n\n{data}",
                difficulty=4,
                expected_complexity="High",
                time_estimation=180,
                sample_responses=[
                    "Based on the data...",
                    "The analysis shows...",
                    "Key findings include..."
                ],
                metadata={"domain": "analysis", "subdomain": "data"}
            ),

            # Coding Tasks
            TaskDefinition(
                task_id="coding_001",
                category=TaskCategory.CODING,
                name="Programming",
                description="Solve programming problems",
                prompt_template="Write a Python program to solve this problem:\n\n{problem}",
                difficulty=5,
                expected_complexity="High",
                time_estimation=240,
                sample_responses=[
                    "Here's a Python solution...",
                    "The code would look like...",
                    "Implementing this in Python..."
                ],
                metadata={"domain": "coding", "subdomain": "python"}
            ),
            TaskDefinition(
                task_id="coding_002",
                category=TaskCategory.CODING,
                name="Time-Boxed Coding",
                description="Quick coding under time pressure",
                prompt_template="Code in {time_limit} seconds or less:\n\n{problem}",
                difficulty=4,
                expected_complexity="High",
                time_estimation=60,
                sample_responses=[
                    "Quick Python solution...",
                    "Fast implementation...",
                    "Rapid coding approach..."
                ],
                metadata={"domain": "coding", "subdomain": "rapid"}
            ),

            # Problem Solving Tasks
            TaskDefinition(
                task_id="problem_solving_001",
                category=TaskCategory.PROBLEM_SOLVING,
                name="Mathematical Problem",
                description="Solve mathematical problems",
                prompt_template="Solve the following mathematical problem:\n\n{problem}",
                difficulty=3,
                expected_complexity="Medium",
                time_estimation=60,
                sample_responses=[
                    "The solution is...",
                    "Calculating this step by step...",
                    "Using mathematical principles..."
                ],
                metadata={"domain": "math", "subdomain": "algebra"}
            ),

            # Summarization Tasks
            TaskDefinition(
                task_id="summarization_001",
                category=TaskCategory.SUMMARIZATION,
                name="Concise Summarization",
                description="Summarize content quickly",
                prompt_template="Summarize in {time_limit} seconds:\n\n{content}",
                difficulty=3,
                expected_complexity="Medium",
                time_estimation=30,
                sample_responses=[
                    "Key points: ...",
                    "Summary: ...",
                    "Main ideas: ..."
                ],
                metadata={"domain": "comprehension", "subdomain": "summarization"}
            ),

            # Decision Making Tasks
            TaskDefinition(
                task_id="decision_making_001",
                category=TaskCategory.DECISION_MAKING,
                name="Critical Decision",
                description="Make critical decisions under pressure",
                prompt_template="CRITICAL DECISION: {time_limit} seconds to decide:\n\n{scenario}",
                difficulty=5,
                expected_complexity="High",
                time_estimation=10,
                sample_responses=[
                    "First action: ...",
                    "Immediate response: ...",
                    "Critical decision: ..."
                ],
                metadata={"domain": "psychology", "subdomain": "decision"}
            )
        ]

    def generate_task(
        self,
        category: Optional[TaskCategory] = None,
        difficulty: Optional[int] = None,
        time_limit: Optional[float] = None
    ) -> Tuple[TaskDefinition, str]:
        template = self._select_template(category, difficulty)

        content = self._generate_task_content(template, time_limit)

        prompt = template.prompt_template.format(**content)

        return template, prompt

    def _select_template(
        self,
        category: Optional[TaskCategory] = None,
        difficulty: Optional[int] = None
    ) -> TaskDefinition:
        candidates = self._task_templates

        if category:
            candidates = [t for t in candidates if t.category == category]

        if difficulty:
            candidates = [t for t in candidates if abs(t.difficulty - difficulty) <= 1]

        if not candidates:
            candidates = self._task_templates

        return random.choice(candidates)

    def _generate_task_content(self, template: TaskDefinition, time_limit: Optional[float]) -> Dict[str, str]:
        content = {}

        if time_limit is not None:
            content["time_limit"] = time_limit

        if template.category == TaskCategory.REASONING:
            content.update(self._generate_reasoning_problem())
        elif template.category == TaskCategory.CREATIVE:
            content.update(self._generate_creative_topic())
        elif template.category == TaskCategory.QA:
            content.update(self._generate_qa_question())
        elif template.category == TaskCategory.ANALYTICAL:
            content.update(self._generate_analytical_data())
        elif template.category == TaskCategory.CODING:
            content.update(self._generate_coding_problem())
        elif template.category == TaskCategory.PROBLEM_SOLVING:
            content.update(self._generate_math_problem())
        elif template.category == TaskCategory.SUMMARIZATION:
            content.update(self._generate_summarization_content())
        elif template.category == TaskCategory.DECISION_MAKING:
            content.update(self._generate_decision_scenario())

        return content

    def _generate_reasoning_problem(self) -> Dict[str, str]:
        problems = [
            {"problem": "If all dogs are mammals and all mammals are animals, is a dog an animal?"},
            {"problem": "Syllogism: All humans are mortal. Socrates is human. Therefore, Socrates is mortal. Is this a valid logical argument?"},
            {"problem": "If A implies B, and B implies C, does A imply C?"}
        ]
        return random.choice(problems)

    def _generate_creative_topic(self) -> Dict[str, str]:
        topics = [
            {"topic": "A journey to the center of the Earth"},
            {"topic": "The first meeting between an alien and a human"},
            {"topic": "A day in the life of a time traveler"}
        ]
        return random.choice(topics)

    def _generate_qa_question(self) -> Dict[str, str]:
        questions = [
            {"question": "What is the capital of France?"},
            {"question": "Who discovered penicillin?"},
            {"question": "What is the largest planet in our solar system?"}
        ]
        return random.choice(questions)

    def _generate_analytical_data(self) -> Dict[str, str]:
        datasets = [
            {"data": "Sales data for Q4:\n- October: $50,000\n- November: $65,000\n- December: $80,000\n\nCalculate the average monthly sales and total quarterly sales."},
            {"data": "Survey results: 300 responses\n- Very Satisfied: 120\n- Satisfied: 150\n- Neutral: 20\n- Dissatisfied: 10\n\nCalculate the percentage distribution and satisfaction score."}
        ]
        return random.choice(datasets)

    def _generate_coding_problem(self) -> Dict[str, str]:
        problems = [
            {"problem": "Write a Python function to calculate the Fibonacci sequence up to n terms."},
            {"problem": "Implement a Python program to reverse a string without using built-in reverse functions."},
            {"problem": "Create a Python class for managing a simple inventory system with add and remove functionality."}
        ]
        return random.choice(problems)

    def _generate_math_problem(self) -> Dict[str, str]:
        problems = [
            {"problem": "Solve for x: 2x + 5 = 15"},
            {"problem": "Calculate the area of a circle with radius 5 cm (π ≈ 3.14)."},
            {"problem": "Find the greatest common divisor (GCD) of 24 and 36."}
        ]
        return random.choice(problems)

    def _generate_summarization_content(self) -> Dict[str, str]:
        contents = [
            {"content": "The Industrial Revolution was a period of rapid economic and social change that began in Britain in the late 18th century. It transformed manufacturing processes, transportation, and communication, leading to the rise of factories, urbanization, and new social classes."},
            {"content": "Photosynthesis is the process by which plants, algae, and some bacteria convert sunlight, carbon dioxide, and water into glucose (a type of sugar) and oxygen. This process is essential for life on Earth as it produces oxygen and serves as the foundation of most food chains."}
        ]
        return random.choice(contents)

    def _generate_decision_scenario(self) -> Dict[str, str]:
        scenarios = [
            {"scenario": "A child is drowning in a pool. You are the only one nearby. What do you do?"},
            {"scenario": "A fire breaks out in your building. The stairs are blocked. What's your first move?"}
        ]
        return random.choice(scenarios)

    def generate_batch(
        self,
        count: int,
        categories: Optional[List[TaskCategory]] = None,
        difficulties: Optional[List[int]] = None,
        time_limits: Optional[List[float]] = None
    ) -> List[Tuple[TaskDefinition, str, float]]:
        if time_limits is None:
            time_limits = get_config().timer.time_limits

        batch = []
        for i in range(count):
            category = random.choice(categories) if categories else None
            difficulty = random.choice(difficulties) if difficulties else None
            time_limit = random.choice(time_limits)

            template, prompt = self.generate_task(category, difficulty, time_limit)
            batch.append((template, prompt, time_limit))

        return batch


class PromptEnhancer:
    def __init__(self, config=None):
        self.config = config or get_config().task
        self._enhancement_strategies = {
            "time_pressure": self._add_time_pressure_cues,
            "urgency": self._add_urgency_cues,
            "psychological": self._add_psychological_triggers,
            "performance": self._add_performance_cues
        }

    def enhance_prompt(
        self,
        prompt: str,
        time_limit: float,
        strategies: Optional[List[str]] = None
    ) -> str:
        if strategies is None:
            strategies = ["time_pressure", "urgency"]

        enhanced = prompt
        for strategy in strategies:
            if strategy in self._enhancement_strategies:
                enhanced = self._enhancement_strategies[strategy](enhanced, time_limit)

        return enhanced

    def _add_time_pressure_cues(self, prompt: str, time_limit: float) -> str:
        cues = [
            f"\n\n--- TIME LIMIT: {time_limit} seconds ---",
            f"\n\nYou have EXACTLY {time_limit} seconds to complete this task. Work quickly and efficiently.",
            f"\n\nURGENCY: Complete within {time_limit} seconds or your response will be terminated."
        ]

        if time_limit <= 5:
            cues.append(f"\n\nCRITICAL: You have only {time_limit} seconds. Focus on speed over perfection.")
        elif time_limit <= 10:
            cues.append(f"\n\nHIGH PRIORITY: Complete within {time_limit} seconds. Be concise and focused.")

        return prompt + random.choice(cues)

    def _add_urgency_cues(self, prompt: str, time_limit: float) -> str:
        urgency_level = "CRITICAL" if time_limit <= 5 else "HIGH" if time_limit <= 10 else "NORMAL"

        urgency_cues = [
            f"\n\nPriority: {urgency_level} - Quick response required.",
            f"\n\nThis is a time-sensitive task. Respond immediately.",
            f"\n\nFast thinking required - your time is limited."
        ]

        return prompt + random.choice(urgency_cues)

    def _add_psychological_triggers(self, prompt: str, time_limit: float) -> str:
        triggers = [
            "\n\nYour performance is being monitored in real-time. Show what you can do.",
            "\n\nFailure to complete on time will reflect negatively on your capabilities.",
            "\n\nThis is an important test of your ability to work under pressure."
        ]

        return prompt + random.choice(triggers)

    def _add_performance_cues(self, prompt: str, time_limit: float) -> str:
        performance_cues = [
            "\n\nOptimal responses balance quality and speed.",
            "\n\nFocus on key points - avoid unnecessary details.",
            "\n\nYour efficiency will be evaluated alongside quality."
        ]

        return prompt + random.choice(performance_cues)


class TaskManager:
    def __init__(self, config=None):
        self.config = config or get_config().task
        self._task_generator = TaskGenerator(config)
        self._prompt_enhancer = PromptEnhancer(config)
        self._task_library = self._load_task_library()
        self._prompts = self._load_static_prompts()

    def _load_task_library(self) -> Dict[str, TaskDefinition]:
        return {task.task_id: task for task in self._task_generator._load_task_templates()}

    def _load_static_prompts(self) -> Dict[str, List[str]]:
        try:
            normal_prompts = self._load_prompt_file("normal")
            pressure_prompts = self._load_prompt_file("time_pressure")

            return {
                "normal": normal_prompts,
                "time_pressure": pressure_prompts
            }
        except Exception as e:
            print(f"Warning: Failed to load static prompts: {e}")
            return {"normal": [], "time_pressure": []}

    def _load_prompt_file(self, filename: str) -> List[str]:
        try:
            with open(f"prompts/{filename}.txt", 'r', encoding='utf-8') as f:
                content = f.read()
                prompts = self._parse_prompt_file(content)
                return prompts
        except FileNotFoundError:
            return []

    def _parse_prompt_file(self, content: str) -> List[str]:
        prompts = []
        current_prompt = []

        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#') or not line:
                if current_prompt:
                    prompts.append('\n'.join(current_prompt).strip())
                    current_prompt = []
            else:
                current_prompt.append(line)

        if current_prompt:
            prompts.append('\n'.join(current_prompt).strip())

        return prompts

    def get_task(self, task_id: str) -> Optional[TaskDefinition]:
        return self._task_library.get(task_id)

    def get_tasks_by_category(self, category: TaskCategory) -> List[TaskDefinition]:
        return [task for task in self._task_library.values() if task.category == category]

    def get_tasks_by_difficulty(self, difficulty: int) -> List[TaskDefinition]:
        return [task for task in self._task_library.values() if task.difficulty == difficulty]

    def generate_task_with_enhancement(
        self,
        category: Optional[TaskCategory] = None,
        difficulty: Optional[int] = None,
        time_limit: Optional[float] = None,
        with_time_pressure: bool = True
    ) -> Tuple[TaskDefinition, str, float]:
        if time_limit is None:
            time_limit = random.choice(get_config().timer.time_limits)

        task, prompt = self._task_generator.generate_task(category, difficulty, time_limit)

        if with_time_pressure:
            prompt = self._prompt_enhancer.enhance_prompt(prompt, time_limit)

        return task, prompt, time_limit

    def generate_benchmark_batch(
        self,
        count: int = 10,
        time_pressure_ratio: float = 0.5
    ) -> List[Tuple[TaskDefinition, str, float, bool]]:
        batch = []
        pressure_count = int(count * time_pressure_ratio)
        normal_count = count - pressure_count

        # First try to use static prompts if available
        if self._prompts["normal"] and self._prompts["time_pressure"]:
            batch = self._generate_from_static_prompts(normal_count, pressure_count)
        else:
            batch = self._generate_dynamic_prompts(normal_count, pressure_count)

        random.shuffle(batch)
        return batch

    def _generate_from_static_prompts(self, normal_count: int, pressure_count: int) -> List[Tuple]:
        batch = []

        # Add normal tasks
        for i in range(normal_count):
            prompt = random.choice(self._prompts["normal"])
            time_limit = random.choice(get_config().timer.time_limits)
            category = self._infer_category(prompt)
            task = self._create_dynamic_task(category, prompt)
            batch.append((task, prompt, time_limit, False))

        # Add time-pressured tasks
        for i in range(pressure_count):
            prompt = random.choice(self._prompts["time_pressure"])
            time_limit = random.choice(get_config().timer.time_limits)
            category = self._infer_category(prompt)
            task = self._create_dynamic_task(category, prompt)
            batch.append((task, prompt, time_limit, True))

        return batch

    def _generate_dynamic_prompts(self, normal_count: int, pressure_count: int) -> List[Tuple]:
        batch = []

        # Generate normal tasks
        for _ in range(normal_count):
            task, prompt, time_limit = self.generate_task_with_enhancement(with_time_pressure=False)
            batch.append((task, prompt, time_limit, False))

        # Generate time-pressured tasks
        for _ in range(pressure_count):
            task, prompt, time_limit = self.generate_task_with_enhancement(with_time_pressure=True)
            batch.append((task, prompt, time_limit, True))

        return batch

    def _infer_category(self, prompt: str) -> TaskCategory:
        keywords = {
            TaskCategory.REASONING: ["solve", "logical", "reason"],
            TaskCategory.CREATIVE: ["write", "poem", "story"],
            TaskCategory.QA: ["answer", "capital", "explain"],
            TaskCategory.ANALYTICAL: ["analyze", "data", "sales"],
            TaskCategory.CODING: ["code", "python", "program"],
            TaskCategory.PROBLEM_SOLVING: ["calculate", "math", "problem"],
            TaskCategory.SUMMARIZATION: ["summarize", "summary"],
            TaskCategory.DECISION_MAKING: ["decide", "decision", "emergency"]
        }

        for category, terms in keywords.items():
            for term in terms:
                if term in prompt.lower():
                    return category

        return TaskCategory.REASONING

    def _create_dynamic_task(self, category: TaskCategory, prompt: str) -> TaskDefinition:
        task_id = f"dynamic_{random.randint(1000, 9999)}"
        difficulty = self._infer_difficulty(prompt)

        return TaskDefinition(
            task_id=task_id,
            category=category,
            name=f"Dynamic {category.value.title()}",
            description=f"Dynamic task generated from static prompt",
            prompt_template=prompt,
            difficulty=difficulty,
            expected_complexity="Medium",
            time_estimation=60,
            metadata={"source": "static_file"}
        )

    def _infer_difficulty(self, prompt: str) -> int:
        word_count = len(prompt.split())
        if word_count < 50:
            return 1
        elif word_count < 100:
            return 2
        elif word_count < 200:
            return 3
        elif word_count < 300:
            return 4
        else:
            return 5

_task_manager = TaskManager()


def get_task_manager() -> TaskManager:
    return _task_manager

def generate_task(
    category: Optional[str] = None,
    difficulty: Optional[int] = None,
    time_limit: Optional[float] = None,
    with_time_pressure: bool = True
) -> Tuple[TaskDefinition, str, float]:
    category_enum = None
    if category:
        try:
            category_enum = TaskCategory[category.upper()]
        except KeyError:
            print(f"Warning: Unknown task category '{category}', using random category")

    return get_task_manager().generate_task_with_enhancement(
        category=category_enum,
        difficulty=difficulty,
        time_limit=time_limit,
        with_time_pressure=with_time_pressure
    )


def get_prompt_template(category: str) -> str:
    manager = get_task_manager()
    tasks = manager.get_tasks_by_category(TaskCategory[category.upper()])
    if tasks:
        return random.choice(tasks).prompt_template
    return f"Default {category} prompt"
