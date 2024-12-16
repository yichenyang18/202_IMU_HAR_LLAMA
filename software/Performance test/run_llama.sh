#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <input_id>"
    exit 1
fi

INPUT_ID="$1"

INPUT_FILE="llama_inputs/inputs_freq_num_samples_${INPUT_ID}.txt"

RESPONSES_FILE="llama_outputs/freq_${INPUT_ID}_responses_prompt.txt"
LOGS_FILE="llama_outputs/freq_${INPUT_ID}_logs_prompt.txt"


> $RESPONSES_FILE
> $LOGS_FILE

OLLAMA_CMD="./ollama run prompt_detail --verbose"

while IFS= read -r query; do
    echo "Processing query: $query"

    echo "Query: $query" >> $RESPONSES_FILE
    echo "Query: $query" >> $LOGS_FILE

    echo "$query" | $OLLAMA_CMD 1>>$RESPONSES_FILE 2>>$LOGS_FILE

    echo "----------------------" >> $RESPONSES_FILE
    echo "----------------------" >> $LOGS_FILE
done < "$INPUT_FILE"

echo "All queries processed."
echo "Responses saved to $RESPONSES_FILE."
echo "Logs saved to $LOGS_FILE."

