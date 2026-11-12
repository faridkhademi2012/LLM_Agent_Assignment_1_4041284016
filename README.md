# LLM Agent Assignment 1

**Student ID:** 4041284016

## Overview

This project implements a single-file LLM-based agent for solving problem-solving tasks using a locally hosted language model through Ollama.

The agent communicates with the local Ollama API and does not require a cloud-based LLM service.

## Requirements

- Python 3.9 or later
- Ollama
- Qwen 2.5 3B model

## Model Setup

Install Ollama from:

https://ollama.com/download

Download the required model:

```bash
ollama pull qwen2.5:3b
```

Make sure the Ollama service is running before executing the program.

## Installation

No additional Python packages are required. The project uses Python standard libraries.

## Input Format

The program receives a JSON input file containing a problem ID and problem description.

Example:

```json
{
  "id": "public_009",
  "problem": "Your problem description goes here."
}
```

## Running the Agent

Run the following command:

```bash
python main.py --input input.json --output output.json --model qwen2.5:3b --think off --max-calls 3 --time-limit 120
```

On Windows, you can also use:

```bash
py main.py --input input.json --output output.json --model qwen2.5:3b --think off --max-calls 3 --time-limit 120
```

## Command-Line Arguments

- `--input`: Input JSON file path.
- `--output`: Output JSON file path.
- `--model`: Ollama model name.
- `--think`: Enable or disable thinking mode.
- `--max-calls`: Maximum number of model calls.
- `--time-limit`: Execution time limit in seconds.
- `--verbose`: Enable detailed execution logs.

## Offline Execution

After installing Ollama and downloading the model, the agent can run without an internet connection, provided the local Ollama service is running.

## Project Structure

```text
LLM_Agent_Assignment_1_4041284016/
│
├── main.py
├── README.md
└── requirements.txt
```

## Output

The agent writes its results to the specified output JSON file.

The output should contain the problem ID and the final answer.

**Note:** All model inference is performed locally using Ollama.
