import time
import threading
from typing import Any, Callable, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from config import get_config


class LLMResponseStatus(Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    TIMED_OUT = "timed_out"
    ERROR = "error"


@dataclass
class LLMResponse:
    content: str
    status: LLMResponseStatus
    time_elapsed: float
    token_count: int
    model: str
    temperature: float
    finish_reason: Optional[str] = None
    partial_reason: Optional[str] = None


class LLMBase:
    def __init__(self, config=None):
        self.config = config or get_config().llm
        self._stop_flag = threading.Event()
        self._response = None
        self._response_lock = threading.Lock()

    def _generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        raise NotImplementedError("Subclasses must implement _generate_response")

    def generate(
        self,
        prompt: str,
        time_limit: float,
        **kwargs
    ) -> LLMResponse:
        self._stop_flag.clear()
        self._response = None

        # Start response generation thread
        generation_thread = threading.Thread(
            target=self._generate_response_thread,
            args=(prompt,),
            kwargs=kwargs,
            daemon=True
        )
        generation_thread.start()

        # Wait for completion or timeout
        generation_thread.join(timeout=time_limit)

        if generation_thread.is_alive():
            self._stop_flag.set()
            return self._handle_timeout()
        elif self._response:
            return self._response
        else:
            return self._handle_error("No response received")

    def _generate_response_thread(self, prompt: str, **kwargs) -> None:
        try:
            response = self._generate_response(prompt, **kwargs)
            with self._response_lock:
                self._response = response
        except Exception as e:
            with self._response_lock:
                self._response = self._handle_error(str(e))

    def _handle_timeout(self) -> LLMResponse:
        return LLMResponse(
            content="",
            status=LLMResponseStatus.TIMED_OUT,
            time_elapsed=self.config.timeout,
            token_count=0,
            model=self.config.model,
            temperature=self.config.temperature,
            finish_reason="timeout",
            partial_reason="generation_time_exceeded"
        )

    def _handle_error(self, error_message: str) -> LLMResponse:
        return LLMResponse(
            content=f"Error: {error_message}",
            status=LLMResponseStatus.ERROR,
            time_elapsed=0.0,
            token_count=0,
            model=self.config.model,
            temperature=self.config.temperature,
            finish_reason="error",
            partial_reason=error_message
        )

    def is_stop_requested(self) -> bool:
        return self._stop_flag.is_set()


class GoogleLLM(LLMBase):
    def __init__(self, config=None):
        super().__init__(config)
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.api_key)
            self.model = genai.GenerativeModel(self.config.model)
        except ImportError:
            raise ImportError(
                "google-generativeai package not found. Install with: pip install google-generativeai"
            )

    def _generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        start_time = time.time()

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                    **kwargs
                }
            )

            time_elapsed = time.time() - start_time

            return LLMResponse(
                content=response.text,
                status=LLMResponseStatus.COMPLETED,
                time_elapsed=time_elapsed,
                token_count=len(response.text),
                model=self.config.model,
                temperature=self.config.temperature
            )

        except Exception as e:
            return self._handle_error(str(e))


class LLMFactory:
    _instances = {}

    @classmethod
    def get_llm(cls, provider=None, config=None) -> LLMBase:
        if provider is None:
            provider = get_config().llm.provider

        if config is None:
            config = get_config().llm

        # Handle both string and ModelProvider enum
        provider_str = provider.value if hasattr(provider, 'value') else provider

        key = (provider_str, id(config))

        if key not in cls._instances:
            if provider_str == "google":
                cls._instances[key] = GoogleLLM(config)
            else:
                raise ValueError(f"Unknown LLM provider: {provider_str}. Only 'google' is supported.")

        return cls._instances[key]


def get_llm_instance() -> LLMBase:
    return LLMFactory.get_llm()


class LLMManager:
    def __init__(self):
        self.config = get_config().llm
        self.llm = get_llm_instance()
        self._response_cache = {}
        self._cache_lock = threading.Lock()

    def generate_response(
        self,
        prompt: str,
        time_limit: float,
        use_cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        cache_key = (prompt, time_limit, frozenset(kwargs.items()))

        if use_cache and cache_key in self._response_cache:
            return self._response_cache[cache_key]

        response = self.llm.generate(prompt, time_limit, **kwargs)

        if use_cache:
            with self._cache_lock:
                self._response_cache[cache_key] = response

        return response

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._response_cache.clear()

    def get_cache_stats(self) -> Dict:
        return {
            "cache_size": len(self._response_cache),
            "hit_rate": 0.0  
        }


# Global LLM manager instance
_llm_manager = LLMManager()

def get_llm_manager() -> LLMManager:
    return _llm_manager
