# api.py — FastAPI wrapper for the fine-tuned QLoRA Mistral-7B Customer Support Model
# Part B: Integration Component (B2)
#
# To run:
#   pip install fastapi uvicorn pydantic torch transformers peft bitsandbytes accelerate
#   uvicorn api:app --host 0.0.0.0 --port 8000
#
# To run on Colab with ngrok (for live demo):
#   See Demo_Notebook.ipynb

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import logging
import time
from datetime import datetime

from config import (
    BASE_MODEL,
    ADAPTER_PATH,
    SYSTEM_PROMPT,
    LOAD_IN_4BIT,
    QUANT_TYPE,
    COMPUTE_DTYPE,
    USE_DOUBLE_QUANT,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    DO_SAMPLE,
    API_HOST,
    API_PORT,
    CORS_ORIGINS,
    LOG_FILE,
)

# Setup logging — logs to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Support LLM API",
    description="QLoRA fine-tuned Mistral-7B for customer support dialogue",
    version="1.0.0",
)

# Allow cross-origin requests (needed for web frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---
class ChatRequest(BaseModel):
    message: str = Field(..., max_length=2000)
    history: Optional[List[dict]] = None
    max_tokens: Optional[int] = Field(default=MAX_NEW_TOKENS, ge=1, le=512)
    temperature: Optional[float] = Field(default=TEMPERATURE, gt=0.0, le=1.0)


class ChatResponse(BaseModel):
    response: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_requests: int
    avg_response_time_ms: float


# Global state
model = None
tokenizer = None
request_count = 0
total_response_time = 0.0


@app.on_event("startup")
async def load_model():
    """Load the fine-tuned model on API startup."""
    global model, tokenizer
    logger.info("Loading base model with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=QUANT_TYPE,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    logger.info(f"Loading fine-tuned adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Model loaded successfully!")


def build_prompt(message: str, history: Optional[List[dict]] = None) -> str:
    """Build Mistral instruct format prompt with optional conversation history."""
    if history:
        prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n"
        for msg in history:
            if msg["role"] == "user":
                prompt += f"Customer: {msg['content']} [/INST] "
            elif msg["role"] == "assistant":
                prompt += f"{msg['content']}</s><s>[INST] "
        prompt += f"Customer: {message} [/INST]"
    else:
        prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\nCustomer: {message} [/INST]"
    return prompt


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate a response to a customer support query."""
    global request_count, total_response_time

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    start_time = time.time()
    request_count += 1

    # Log incoming request
    logger.info(
        f"Request #{request_count} | Message: {request.message[:100]}... "
        f"| History turns: {len(request.history) if request.history else 0}"
    )

    try:
        prompt = build_prompt(request.message, request.history)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=TOP_P,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        elapsed = (time.time() - start_time) * 1000
        total_response_time += elapsed

        # Log successful response
        logger.info(
            f"Request #{request_count} | Response time: {elapsed:.0f}ms "
            f"| Response length: {len(response_text)} chars"
        )

        return ChatResponse(
            response=response_text,
            processing_time_ms=round(elapsed, 2),
        )

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        logger.error(
            f"Request #{request_count} | FAILED after {elapsed:.0f}ms | Error: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint with monitoring stats."""
    avg_time = (total_response_time / request_count) if request_count > 0 else 0.0
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        total_requests=request_count,
        avg_response_time_ms=round(avg_time, 2),
    )
