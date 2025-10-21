#!/bin/bash
set -euo pipefail


export MODEL_NAME="unsloth/gemma-3-27b-it-GGUF:Q4_K_M"
export MODEL_URL="http://127.0.0.1:8080/v1/chat/completions"
export OPENAI_API_KEY="${OPENAI_API_KEY:-placeholder}"

# bash ./scripts/llm_new_pipeline.sh torch openai # use it at the first time
bash ./scripts/llm_pipeline.sh torch openai # use it to resume from break

