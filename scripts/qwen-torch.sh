#!/bin/bash
set -euo pipefail


export MODEL_NAME="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_XL"
export MODEL_URL="http://127.0.0.1:8080/v1/chat/completions"
export OPENAI_API_KEY="${OPENAI_API_KEY:-placeholder}"

bash ./scripts/llm_new_pipeline.sh torch openai # use it at the first time
# bash ./scripts/llm_pipeline.sh torch openai # use it to resume from break

