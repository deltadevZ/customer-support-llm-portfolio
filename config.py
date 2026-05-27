# config.py — Centralised configuration for the Customer Support LLM
# All hardcoded values extracted here for easy modification and deployment

# --- Model Configuration ---
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH = "./model/final"  # Path to saved LoRA adapter

# --- Quantization ---
LOAD_IN_4BIT = True
QUANT_TYPE = "nf4"
COMPUTE_DTYPE = "float16"
USE_DOUBLE_QUANT = True

# --- LoRA Configuration ---
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# --- Training Configuration ---
MAX_STEPS = 750
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
WARMUP_STEPS = 50
LR_SCHEDULER = "cosine"
MAX_LENGTH = 512
PACKING = True
SEED = 42

# --- Inference Configuration ---
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# --- API Configuration ---
API_HOST = "0.0.0.0"
API_PORT = 8000
LOG_FILE = "api_requests.log"
CORS_ORIGINS = ["*"]

# --- System Prompt ---
SYSTEM_PROMPT = """You are a helpful customer support assistant. You provide clear,
accurate, and professional responses to customer inquiries. Be concise but thorough,
and always maintain a polite and supportive tone."""
