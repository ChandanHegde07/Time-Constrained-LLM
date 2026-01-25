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


class OpenAILLM(LLMBase):
    def __init__(self, config=None):
        super().__init__(config)
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
        except ImportError:
            raise ImportError(
                "openai package not found. Install with: pip install openai"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    def _generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        start_time = time.time()

        try:
            # Configure API parameters
            params = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
                "timeout": self.config.timeout,
                **kwargs
            }

            # Stream response if available
            if self.config.enable_streaming:
                response = self._generate_streaming_response(prompt, params)
            else:
                response = self._generate_batch_response(prompt, params)

            time_elapsed = time.time() - start_time

            return LLMResponse(
                content=response["content"],
                status=LLMResponseStatus.COMPLETED,
                time_elapsed=time_elapsed,
                token_count=response["token_count"],
                model=self.config.model,
                temperature=self.config.temperature,
                finish_reason=response["finish_reason"]
            )

        except Exception as e:
            return self._handle_error(str(e))

    def _generate_batch_response(self, prompt: str, params: Dict) -> Dict:
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                **params
            )

            return {
                "content": completion.choices[0].message.content or "",
                "token_count": completion.usage.total_tokens,
                "finish_reason": completion.choices[0].finish_reason
            }

        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    def _generate_streaming_response(self, prompt: str, params: Dict) -> Dict:
        try:
            response_chunks = []
            token_count = 0
            finish_reason = None

            stream = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **params
            )

            for chunk in stream:
                if self.is_stop_requested():
                    finish_reason = "stop"
                    break

                if chunk.choices[0].delta.content:
                    response_chunks.append(chunk.choices[0].delta.content)
                    token_count += 1

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                    break

            return {
                "content": "".join(response_chunks),
                "token_count": token_count,
                "finish_reason": finish_reason
            }

        except Exception as e:
            raise RuntimeError(f"OpenAI streaming error: {e}")


class AnthropicLLM(LLMBase):
    def __init__(self, config=None):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=self.config.api_key
            )
        except ImportError:
            raise ImportError(
                "anthropic package not found. Install with: pip install anthropic"
            )

    def _generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        start_time = time.time()

        try:
            completion = self.client.completions.create(
                model=self.config.model,
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
                max_tokens_to_sample=self.config.max_tokens,
                temperature=self.config.temperature,
                **kwargs
            )

            time_elapsed = time.time() - start_time

            return LLMResponse(
                content=completion.completion,
                status=LLMResponseStatus.COMPLETED,
                time_elapsed=time_elapsed,
                token_count=completion.stop_sequence,
                model=self.config.model,
                temperature=self.config.temperature,
                finish_reason=completion.stop_reason
            )

        except Exception as e:
            return self._handle_error(str(e))


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


class LocalLLM(LLMBase):
    def __init__(self, config=None):
        super().__init__(config)
        import random
        self.random = random.Random()

    def _generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        start_time = time.time()

        try:
            # Simulate response generation time
            response_time = self.random.uniform(0.5, self.config.timeout * 0.8)
            time.sleep(response_time)

            if self.is_stop_requested():
                return LLMResponse(
                    content="This is a partial response...",
                    status=LLMResponseStatus.PARTIAL,
                    time_elapsed=time.time() - start_time,
                    token_count=100,
                    model=self.config.model,
                    temperature=self.config.temperature,
                    finish_reason="stop",
                    partial_reason="time_limit_exceeded"
                )

            # Generate simulated response
            response = f"Generated response to: {prompt}"

            time_elapsed = time.time() - start_time

            return LLMResponse(
                content=response,
                status=LLMResponseStatus.COMPLETED,
                time_elapsed=time_elapsed,
                token_count=len(response),
                model=self.config.model,
                temperature=self.config.temperature,
                finish_reason="stop"
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

        key = (provider, id(config))

        if key not in cls._instances:
            if provider == "openai":
                cls._instances[key] = OpenAILLM(config)
            elif provider == "anthropic":
                cls._instances[key] = AnthropicLLM(config)
            elif provider == "google":
                cls._instances[key] = GoogleLLM(config)
            elif provider == "local":
                cls._instances[key] = LocalLLM(config)
            else:
                raise ValueError(f"Unknown LLM provider: {provider}")

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
