import LiteralMessagePassing as lmp
import argparse
import asyncio
import datetime
import os
import json
import random
import networkx as nx
from networkx.readwrite import json_graph
import subprocess
import pandas as pd
from datasets import load_dataset


TASKS = {
    "matching": lmp.Matching,
    "consensus": lmp.Consensus,
    "coloring": lmp.Coloring,
    "leader_election": lmp.LeaderElection,
    "vertex_cover": lmp.VertexCover,
}

MODEL_PROVIDER = {
    "gpt-4o-mini": "openai",
    "gpt-4.1-mini": "openai",
    "gpt-4o": "openai",
    "o1": "openai",
    "o3-mini": "openai",
    "o4-mini": "openai",
    "llama3.1": "ollama",
    "gemini-2.0-flash": "google-genai",
    "gemini-2.0-flash-lite": "google-genai",
    "claude-3-5-haiku-20241022": "anthropic",
    "claude-3-opus-20240229": "anthropic",
    "claude-3-7-sonnet-20250219": "anthropic",
    "claude-3-7-sonnet-20250219-thinking": "anthropic",
    "gemini-2.5-flash-preview-04-17": "google-genai",
    "gemini-2.5-flash-preview-04-17-thinking": "google-genai",
    "gemini-2.5-pro-exp-03-25": "google-genai",
    "gemini-2.5-pro-preview-03-25": "google-genai",
    "gemini-2.5-pro-preview-05-06": "google-genai",
    "gemini-1.5-pro": "google-genai",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "together",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "together"
}


def get_graph(graph_model, graph_size, num_sample):
    dataset = load_dataset("disco-eth/AgentsNet", split="train")
    _loaded_hf_df = pd.DataFrame(dataset)

    row = _loaded_hf_df[
        (_loaded_hf_df["graph_generator"] == graph_model) &
        (_loaded_hf_df["num_nodes"] == graph_size) &
        (_loaded_hf_df["index"] == num_sample)
    ]

    if len(row) == 0:
        raise ValueError(f"Graph not found: {graph_model}_{graph_size}_{num_sample}")

    graph_dict = json.loads(row.iloc[0]["graph"])
    print(f"Loaded graph from Hugging Face: {graph_model}_{graph_size}_{num_sample}")
    return json_graph.node_link_graph(graph_dict["graph"], edges="links")

def get_git_commit_hash():
    '''This function is failsafe even if git is not installed on the system.'''
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        return commit_hash
    except Exception as e:
        return "None"

def save_results(answers, transcripts, graph, rounds, model_name, task, score, commit_hash, graph_generator, graph_index, successful, error_message, chain_of_thought, num_fallbacks, num_failed_json_parsings_after_retry, num_failed_answer_parsings_after_retry):
    """Saves the experiment results and message transcripts to a JSON file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task}_results_{timestamp}_rounds{rounds}_{model_name.split('/')[-1]}_nodes{len(graph.nodes())}.json"

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump({
            'answers': answers,
            'transcripts': transcripts,
            'graph': json_graph.node_link_data(graph),
            'num_nodes': len(graph.nodes()),
            'diameter': nx.diameter(graph),
            'max_degree': max(dict(graph.degree()).values()),
            'rounds': rounds,
            'model_name': model_name,
            'task': task,
            'score': score,
            'commit_hash': commit_hash,
            'graph_generator': graph_generator,
            'graph_index': graph_index,
            'successful': successful,
            'error_message': error_message,
            'chain_of_thought': chain_of_thought,
            'num_fallbacks': num_fallbacks,
            'num_failed_json_parsings_after_retry': num_failed_json_parsings_after_retry,
            'num_failed_answer_parsings_after_retry': num_failed_answer_parsings_after_retry,
        }, f, indent=4)

    print(f"Results saved to {filepath}")


LOCAL_ROUNDS = {
    10: [4, 6, 8],
    20: [6, 8, 10],
    50: [8, 10], 
}


def determine_rounds(task, graph, num_sample, num_samples, rounds):
    if task in ["consensus", "leader_election"] or graph.number_of_nodes() > 16:
        return 2 * nx.diameter(graph) + 1
    else:
        return rounds


async def run(args):
    results = []
    commit_hash = get_git_commit_hash()
    random.seed(args.seed)

    if args.missing_run_file is not None:
        recovery_mode = True
        missing_run_df = pd.read_csv(args.missing_run_file)
    else:
        recovery_mode = False

    for graph_model in args.graph_models:
        for i in range(args.start_from_sample, args.samples_per_graph_model):
            graph = get_graph(graph_model, args.graph_size, i)
            rounds = determine_rounds(args.task, graph, i, args.samples_per_graph_model, args.rounds)
            print(f"Selected {rounds} rounds.")

            task_class = TASKS[args.task]
            model_provider = MODEL_PROVIDER[args.model]
            chain_of_thought = not args.disable_chain_of_thought

            runs_to_execute = 1
            if recovery_mode:
                print('### Entered recovery mode')
                graph_string = str(json_graph.node_link_data(graph))
                filtered_df = missing_run_df[(missing_run_df.num_nodes == len(graph.nodes)) & (missing_run_df.task == args.task) & (missing_run_df.graph_generator == graph_model) & (missing_run_df.graph == graph_string) & (missing_run_df.model_name == args.model)]
                if len(filtered_df) > 0:
                    print('### Running recovery!')
                    runs_to_execute = filtered_df.iloc[0].missing_runs
                else:
                    print('### Skipping run!')
                    runs_to_execute = 0

            for _ in range(runs_to_execute):
                lmp_model: lmp.LiteralMessagePassing = task_class(graph=graph, rounds=rounds, model_name=args.model, model_provider=model_provider, chain_of_thought=chain_of_thought)
                await lmp_model.bootstrap()
                try:
                    answers = await lmp_model.pass_messages()
                    score = lmp_model.get_score(answers)
                    successful = True
                    error_message = None
                except (ValueError, KeyError) as e:
                    answers = [None for _ in range(graph.order())]
                    score = None
                    successful = False
                    error_message = repr(e)

                results.append(dict(model=args.model, task=args.task, rounds=rounds, seed=args.seed, score=score))
                save_results(
                    answers=answers,
                    transcripts=lmp_model.get_transcripts(),
                    graph=lmp_model.graph,
                    rounds=rounds,
                    model_name=lmp_model.model_name,
                    task=args.task,
                    score=score,
                    commit_hash=commit_hash,
                    graph_generator=graph_model,
                    graph_index=i,
                    successful=successful,
                    error_message=error_message,
                    chain_of_thought=chain_of_thought,
                    num_fallbacks = lmp_model.num_fallbacks,
                    num_failed_json_parsings_after_retry = lmp_model.num_failed_json_parsings_after_retry,
                    num_failed_answer_parsings_after_retry = lmp_model.num_failed_answer_parsings_after_retry
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--task", type=str, default="coloring")
    parser.add_argument("--graph_models", type=str, nargs="+", default=["ws", "ba", "dt"])
    parser.add_argument("--start_from_sample", type=int, default=0)
    parser.add_argument("--samples_per_graph_model", type=int, default=3)
    parser.add_argument("--graph_size", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_chain_of_thought", action="store_true")
    parser.add_argument("--missing_run_file", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(run(args))
