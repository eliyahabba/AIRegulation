# Task Generation Constants (for creating prompt variations)
TASK_DEFAULT_VARIATIONS_PER_FIELD = 10
TASK_DEFAULT_MAX_VARIATIONS_PER_ROW = 10
TASK_DEFAULT_MAX_ROWS = 20
TASK_DEFAULT_RANDOM_SEED = 42
TASK_DEFAULT_PLATFORM = "TogetherAI"
TASK_DEFAULT_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
# TASK_DEFAULT_PLATFORM = "OpenAI"
# TASK_DEFAULT_MODEL_NAME = "gpt-4o-mini"

# Language Model Running Constants (for running models on variations)
LM_DEFAULT_MAX_TOKENS = 1024
LM_DEFAULT_PLATFORM = "TogetherAI"
LM_DEFAULT_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
LM_DEFAULT_TEMPERATURE = 0.0
LM_DEFAULT_PARALLEL_WORKERS = 6  # Number of parallel workers for model calls (1 = sequential)
LM_DEFAULT_MAX_RETRIES = 3  # Maximum number of retries for rate limit errors
LM_DEFAULT_RETRY_SLEEP = 60  # Base sleep time in seconds for rate limit retries
LM_DEFAULT_BATCH_SIZE = 50  # Number of variations to process before saving intermediate results
LM_DEFAULT_REQUIRE_GPU = True  # Whether to require GPU for local models

# Platform options
PLATFORMS = {
    "TogetherAI": "TogetherAI",
    "OpenAI": "OpenAI",
    "local": "local"
}

# Model names by platform
MODELS = {
    "TogetherAI": {
        "default": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "llama_3_3_70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "OpenAI": {
        "default": "gpt-4o-mini",
        "gpt_4o_mini": "gpt-4o-mini",
        "gpt_4o": "gpt-4o",
    },
    "local": {
        "default": "meta-llama/Llama-2-7b-chat-hf",
        "llama2_7b": "meta-llama/Llama-2-7b-chat-hf",
        "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.2",
        "qwen_7b": "Qwen/Qwen2-7B-Instruct",
        "gemma_7b": "google/gemma-7b-it",
        "phi_3_5": "microsoft/Phi-3.5-mini-4k-instruct",
        "deepseek_7b": "deepseek-ai/deepseek-coder-7b-instruct",
        "vicuna_7b": "lmsys/vicuna-7b-v1.5",
        "falcon_7b": "tiiuae/falcon-7b-instruct",
        "mpt_7b": "mosaicml/mpt-7b-instruct",
    }
}

# Short model names for file naming
MODEL_SHORT_NAMES = {
    # TogetherAI models
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": "llama_3_3_70b_free",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "llama_3_3_70b",
    
    # OpenAI models
    "gpt-4o-mini": "gpt_4o_mini",
    "gpt-4o": "gpt_4o",
    
    # Local models
    "meta-llama/Llama-2-7b-chat-hf": "llama2_7b",
    "mistralai/Mistral-7B-Instruct-v0.2": "mistral_7b",
    "Qwen/Qwen2-7B-Instruct": "qwen_7b",
    "google/gemma-7b-it": "gemma_7b",
    "microsoft/Phi-3.5-mini-4k-instruct": "phi_3_5",
    "deepseek-ai/deepseek-coder-7b-instruct": "deepseek_7b",
    "lmsys/vicuna-7b-v1.5": "vicuna_7b",
    "tiiuae/falcon-7b-instruct": "falcon_7b",
    "mosaicml/mpt-7b-instruct": "mpt_7b",
}

# Backward compatibility aliases (deprecated - use specific TASK_ or LM_ prefixed constants)
DEFAULT_VARIATIONS_PER_FIELD = TASK_DEFAULT_VARIATIONS_PER_FIELD
DEFAULT_MAX_VARIATIONS_PER_ROW = TASK_DEFAULT_MAX_VARIATIONS_PER_ROW
DEFAULT_MAX_ROWS = TASK_DEFAULT_MAX_ROWS
DEFAULT_RANDOM_SEED = TASK_DEFAULT_RANDOM_SEED
DEFAULT_PLATFORM = TASK_DEFAULT_PLATFORM
DEFAULT_MODEL_NAME = TASK_DEFAULT_MODEL_NAME
DEFAULT_MAX_TOKENS = LM_DEFAULT_MAX_TOKENS
DEFAULT_PARALLEL_WORKERS = LM_DEFAULT_PARALLEL_WORKERS
DEFAULT_MAX_RETRIES = LM_DEFAULT_MAX_RETRIES
DEFAULT_RETRY_SLEEP = LM_DEFAULT_RETRY_SLEEP
DEFAULT_BATCH_SIZE = LM_DEFAULT_BATCH_SIZE
DEFAULT_REQUIRE_GPU = LM_DEFAULT_REQUIRE_GPU




class GenerationDefaults:
    """Centralized defaults for generation parameters across API, CLI, and UI."""
    MAX_VARIATIONS_PER_ROW = None  # None means no limit on variations
    MAX_ROWS = None  # None means use all rows
    VARIATIONS_PER_FIELD = 3
    MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    API_PLATFORM = "TogetherAI"
    RANDOM_SEED = 42
