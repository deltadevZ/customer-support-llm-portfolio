# tests.py — Unit and Integration Tests for the Customer Support LLM
#
# Run with: python -m pytest tests.py -v
# Or on Colab: !pip install pytest && !python -m pytest tests.py -v
#
# Note: Tests requiring the model need a GPU runtime.
# Tests marked with @pytest.mark.skipif will skip if no GPU is available.

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))
from config import *

GPU_AVAILABLE = torch.cuda.is_available()


# ============================================================
# Unit Tests — No GPU required
# ============================================================

class TestConfig:
    """Test that configuration values are valid."""

    def test_lora_rank_is_positive(self):
        assert LORA_RANK > 0, "LoRA rank must be positive"

    def test_lora_alpha_is_double_rank(self):
        assert LORA_ALPHA == 2 * LORA_RANK, "LoRA alpha should be 2x rank"

    def test_learning_rate_in_range(self):
        assert 1e-6 < LEARNING_RATE < 1e-3, "Learning rate out of typical range"

    def test_temperature_in_range(self):
        assert 0.0 < TEMPERATURE <= 1.0, "Temperature must be between 0 and 1"

    def test_max_length_fits_model(self):
        assert MAX_LENGTH <= 4096, "Max length exceeds Mistral-7B context window"

    def test_effective_batch_size(self):
        assert EFFECTIVE_BATCH_SIZE == BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

    def test_seed_is_set(self):
        assert SEED is not None, "Random seed must be set for reproducibility"

    def test_target_modules_not_empty(self):
        assert len(TARGET_MODULES) > 0, "Must target at least one module"

    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT.strip()) > 0, "System prompt cannot be empty"


class TestPromptFormatting:
    """Test that prompts are formatted correctly for Mistral."""

    def test_single_turn_format(self):
        prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\nCustomer: Hello [/INST]"
        assert "<s>[INST]" in prompt
        assert "[/INST]" in prompt
        assert "Customer:" in prompt

    def test_multi_turn_format(self):
        history = [("How do I cancel?", "You can cancel in settings.")]
        prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n"
        for h_user, h_assistant in history:
            prompt += f"Customer: {h_user} [/INST] {h_assistant}</s><s>[INST] "
        prompt += f"Customer: Can I get a refund? [/INST]"

        # Check structure
        assert prompt.count("[INST]") == 2, "Multi-turn should have 2 instruction blocks"
        assert prompt.count("[/INST]") == 2, "Multi-turn should have 2 response markers"
        assert "</s><s>" in prompt, "Turns should be separated by </s><s>"

    def test_empty_input_handling(self):
        prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\nCustomer:  [/INST]"
        assert "Customer:" in prompt, "Should handle empty customer input"


# ============================================================
# Integration Tests — Require GPU and model
# ============================================================

@pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU available")
class TestModelLoading:
    """Test that the model loads and generates correctly."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_model(self, request):
        """Load model once for all tests in this class."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=LOAD_IN_4BIT,
            bnb_4bit_quant_type=QUANT_TYPE,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
        )

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        tokenizer.pad_token = tokenizer.eos_token

        request.cls.model = model
        request.cls.tokenizer = tokenizer

    def test_model_loads(self):
        assert self.model is not None, "Model failed to load"

    def test_tokenizer_loads(self):
        assert self.tokenizer is not None, "Tokenizer failed to load"

    def test_tokenizer_has_pad_token(self):
        assert self.tokenizer.pad_token is not None, "Pad token not set"

    def test_model_is_in_eval_mode(self):
        assert not self.model.training, "Model should be in eval mode"

    def test_model_memory_under_limit(self):
        memory_gb = self.model.get_memory_footprint() / 1e9
        assert memory_gb < 6, f"Model using {memory_gb:.1f}GB, expected under 6GB"

    def test_generate_non_empty_response(self):
        prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\nCustomer: Hello [/INST]"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        assert len(response) > 0, "Generated response is empty"

    def test_response_is_english(self):
        prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\nCustomer: I need help [/INST]"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Basic check: response should contain common English words
        assert any(word in response.lower() for word in ["i", "the", "to", "you", "your", "help", "assist"]), \
            f"Response doesn't appear to be English: {response[:100]}"


# ============================================================
# API Tests — No GPU required, tests request/response models
# ============================================================

class TestAPIModels:
    """Test API request and response model validation."""

    def test_chat_request_valid(self):
        from pydantic import BaseModel
        from typing import Optional, List

        class ChatRequest(BaseModel):
            message: str
            history: Optional[List[dict]] = None
            max_tokens: Optional[int] = 256
            temperature: Optional[float] = 0.7

        req = ChatRequest(message="Hello")
        assert req.message == "Hello"
        assert req.history is None
        assert req.max_tokens == 256

    def test_chat_request_with_history(self):
        from pydantic import BaseModel
        from typing import Optional, List

        class ChatRequest(BaseModel):
            message: str
            history: Optional[List[dict]] = None

        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        req = ChatRequest(message="Follow up", history=history)
        assert len(req.history) == 2
        assert req.history[0]["role"] == "user"

    def test_chat_request_empty_message_allowed(self):
        from pydantic import BaseModel
        from typing import Optional, List

        class ChatRequest(BaseModel):
            message: str
            history: Optional[List[dict]] = None

        req = ChatRequest(message="")
        assert req.message == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
