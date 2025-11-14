touch .tmp/result.csv
printf "LLM,Metric,Reference,Current,Difference\n" > .tmp/result.csv
call() {
    lib=$1
    llm=$2
    python utils/compare_results.py .tmp/${lib}-gemini .tmp/${lib}-${llm} gemini ${llm} ${lib} >> .tmp/result.csv
}

call torch openai
call torch claude
call torch gemma
call torch qwen
call tf openai
call tf gemma
call tf qwen