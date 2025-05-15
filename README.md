# AgentsNet

Code for our submission.

## Installation

To set up the project, follow these steps:

### 1. Create a Conda Environment

```bash
conda create -n agentsnet python=3.11 -y
conda activate agentsnet
```
### 2. Install Dependencies
Install the required Python packages:
```bash
pip install datasets langchain langgraph langchain-openai langchain-ollama langchain-google-genai langchain-anthropic pandas scipy networkx numpy==1.26.4
```

## Running experiments
First make sure that you set the API key for the provider you want to use. For google genai, this would be:
```bash
export GOOGLE_API_KEY=<INSERT-API-KEY-HERE>
```
For openai:
```bash
export OPENAI_API_KEY=<INSERT-API-KEY-HERE>
```
For anthropic:
```bash
export ANTHROPIC_API_KEY=<INSERT-API-KEY-HERE>
```
Then, you can start a run:
```bash
python main.py --graph_size 16 --task coloring --rounds 8 --samples_per_graph_model 3 --model gemini-2.0-flash 
```
This runs 12 instances of the coloring task of 16 nodes for 8 rounds, with 4 different graph classes and 3 samples per graph class each, with gemini-2.0-flash as the model. See `main.sh` to run a complete run of the benchmark for one particular model.

## Running more extensive experiments
To run the AgentsNet benchmark on all tasks with 12 graphs each (of size 16), you can use:
```bash
./main.sh gemini-2.0-flash
```
This will run with gemini-2.0-flash. You can choose from a series of OpenAI, Anthropic and Gemini models.

## Running the chat tool
We have a simple chat tool to read transcripts from completed runs. Simply run
```
python chat_tool.py --file [FILE]
```
where `[FILE]` is a results file produced by `main.py`. Providing `--agents agent1 agent2` will print the chat between agent1 and agent2. Only providing `--agents agent1` will print the transcript of agent1.

## Dataset

All graph instances used in our benchmark are publicly available on the Hugging Face Hub: [https://huggingface.co/datasets/disco-eth/AgentsNet](https://huggingface.co/datasets/disco-eth/AgentsNet)

The dataset consists of synthetic graphs generated using various random graph models. It serves as the input for all experiments in the benchmark.

