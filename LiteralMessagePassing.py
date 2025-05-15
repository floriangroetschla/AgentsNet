from abc import ABC
import asyncio
import networkx as nx
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
import json
import os
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_ollama import ChatOllama
from langchain.schema import messages_to_dict
import re
import regex
from utils import names

def parse_messages(response: str) -> dict[str, str]:
    response = response.replace("\\n", "\n")
    pattern = r'(\{.*\})'
    matches = regex.findall(pattern, response, regex.DOTALL, overlapped=True)

    if not matches:
        return None

    for json_candidate in matches:
        try:
            out = json.loads(json_candidate)
            if all([k in names for k in out.keys()]) and all([isinstance(v, str) for v in out.values()]):
                return out
        except json.JSONDecodeError:
            continue

    return None


RATE_LIMITER_KWARGS = {
    "gpt-4o-mini": dict(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=200,
    ),
    "gpt-4.1-mini": dict(
        requests_per_second=5,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": dict(
        requests_per_second=2,
        check_every_n_seconds=0.1,
        max_bucket_size=4,
    ),
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": dict(
        requests_per_second=50,
        check_every_n_seconds=0.1,
        max_bucket_size=100,
    ),
    "gpt-4o": dict(
        requests_per_second=1,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "o1": dict(
        requests_per_second=1,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "o1-mini": dict(
        requests_per_second=1,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),

    "o3-mini": dict(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "o4-mini": dict(
        requests_per_second=5,
        check_every_n_seconds=0.2,
        max_bucket_size=20,
    ),
    "llama3.1": dict(
        requests_per_second=20,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "gemini-2.0-flash": dict(
        requests_per_second=100,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "gemini-2.0-flash-lite": dict(
        requests_per_second=100,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "claude-3-5-haiku-20241022": dict(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "claude-3-opus-20240229": dict(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "claude-3-7-sonnet-20250219": dict(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "claude-3-7-sonnet-20250219-thinking": dict(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "gemini-2.5-pro-exp-03-25": dict(
        requests_per_second=2,
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    ),
    "gemini-2.5-pro-preview-03-25": dict(
        requests_per_second=2,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    ),
    "gemini-2.5-pro-preview-05-06": dict(
        requests_per_second=2,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    ),
    "gemini-2.5-flash-preview-04-17": dict(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "gemini-2.5-flash-preview-04-17-thinking": dict(
        requests_per_second=10,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
    "gemini-1.5-pro": dict(
        requests_per_second=100,
        check_every_n_seconds=0.1,
        max_bucket_size=20,
    ),
}


class LiteralMessagePassing(ABC):
    def __init__(self, graph, model_name="gpt-4o", model_provider="openai", chain_of_thought=True):
        self.graph = graph
        self.model_name = model_name
        self.chain_of_thought = chain_of_thought
        rate_limiter = InMemoryRateLimiter(**RATE_LIMITER_KWARGS[model_name])
        if model_provider == 'ollama':
            self.model = ChatOllama(model=model_name, base_url=os.environ['OLLAMA_URI'])
        else:
            chat_kwargs = {}

            if model_name.startswith("claude-3-7-sonnet"):
                if model_name.endswith("thinking"):
                    model_name = "-".join(model_name.split("-")[:-1])
                    chat_kwargs.update({
                        "max_tokens": 5000,
                        "thinking": {
                            "type": "enabled",
                            "budget_tokens": 2000 
                        }
                    })
            if model_name == "gemini-2.5-flash-preview-04-17-thinking":
                model_name = "gemini-2.5-flash-preview-04-17"
                chat_kwargs.update({
                    "thinking_budget": 24000,  # max for Gemini 2.5 Flash
                    "include_thoughts": True,
                })

            self.model = init_chat_model(model_name, model_provider=model_provider, rate_limiter=rate_limiter, **chat_kwargs).with_retry(stop_after_attempt=10)
        self.workflow = StateGraph(state_schema=MessagesState)
        self.transcripts = []

        def call_model(state: MessagesState):
            response = self.model.invoke(state["messages"])
            return {"messages": response}
            
        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", call_model)

        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

        self.messages = {v: [] for v in graph.nodes()}
        self.chat_history = {v: [] for v in graph.nodes()}
        self.num_fallbacks = [0 for v in graph.nodes()]
        self.num_failed_json_parsings_after_retry = [0 for v in graph.nodes()]
        self.num_failed_answer_parsings_after_retry = [0 for v in graph.nodes()]

        self.bootstrap_template = PromptTemplate.from_template(
            """You are an agent that is connected with other agents (your neighbors), who you communicate with. Your neighbors can in turn communicate with their neighbors and so forth. {short_task_description}
The rules are as follows:
1. There are {num_agents} agents in total. Everybody has a unique name. Your name is {own_name}.
2. You can only communicate with your immediate neighbors ({neighbors}). You cannot see or directly communicate with anyone else, unless information is relayed by intermediate agents.
3. You can exchange text-based messages with your neighbors in rounds. In each round, you will first receive the last messages sent by your neighbors and then be asked to generate your response messages which your neighbors receive in the next round. This process repeats for {num_rounds} rounds of message passing. Importantly, the process is synchronous: Every agent decides on which messages to send at the same time and sees the messages from other agents only in the next round.
4. Everybody (including you) decides what to share or request from neighbors. In every round, think step-by-step about the next set of messages you want to send. Output a JSON string that contains your response messages.
5. The messages you send to your neighbors are formatted as JSON. For example, if your neighbors are Alan and Bob, your output should look as follows: 
```
{{
    "Alan": "Message that will be sent to Alan.",
    "Bob": "Message that will be sent to Bob.",
}}
```
It is not mandatory to send a message to every neighbor in every round. If you do not want to send a message to a particular neighbor, you may omit their name from the JSON.
6. After {num_rounds} message passes, you have to solve the following task: {long_problem_description}"""
        )

        self.cot_prompt = "Elaborate your chain of thought step-by-step first, then output the messages for your neighbors."
        self.cot_prompt_final_prediction = "\n\nElaborate your chain of thought step-by-step first, then answer the following: "
        if not chain_of_thought:
            self.cot_prompt = ""
            self.cot_prompt_final_prediction = ""

        self.bootstrap_ask_for_first_messages = PromptTemplate.from_template("What are the first messages you want to send to your neighbors? {cot_prompt} Output your messages in JSON format as specified earlier.")
        self.format_instructions = PromptTemplate.from_template("{question} Format your answer as follows: '### Final Answer ###', followed by your final answer. Don't use any text for your final answer except one of these valid options: {valid_answers}")

    async def fallback_json_request(self, node_id):
        self.num_fallbacks[node_id] += 1
        user_message = HumanMessage(content="Your messages could not be parsed into JSON. Please check your response and try again.")
        config = {"configurable": {"thread_id": str(node_id)}}
        response = await self.app.ainvoke({'messages': [user_message]}, config=config)
        self.chat_history[node_id] = messages_to_dict(response['messages'])
        last_message = self.chat_history[node_id][-1]['data']['content']
        messages_sent = parse_messages(last_message)
        if messages_sent is None:
            self.num_failed_json_parsings_after_retry[node_id] += 1
            return {}
        return messages_sent
    
    async def fallback_answer_request(self, node_id):
        self.num_fallbacks[node_id] += 1
        user_message = HumanMessage(content="Your answer could not be parsed. Please check your answer and try again.")
        config = {"configurable": {"thread_id": str(node_id)}}
        response = await self.app.ainvoke({'messages': [user_message]}, config=config)
        self.chat_history[node_id] = messages_to_dict(response['messages'])
        last_message = self.chat_history[node_id][-1]['data']['content']
        messages_sent = self.parse_answer(node_id, last_message)
        if messages_sent is None:
            self.num_failed_answer_parsings_after_retry[node_id] += 1
            return None
        return messages_sent

    def parse_answer(self, node, message):
        valid_answers = self.get_valid_answers(node)
        pattern = r"### Final Answer ###\s*(" + "|".join(re.escape(ans) for ans in valid_answers) + r")"
        parsed_answer = re.search(pattern, message)
        if parsed_answer:
            return parsed_answer.group(1)
        return None

    async def parse_response_to_dict(self, message_dict, node_id, last_round=False):
        last_message = message_dict[-1]['data']['content']
        if not last_round:
            messages_sent = parse_messages(last_message)
        else:
            messages_sent = self.parse_answer(node_id, last_message)

        if messages_sent is None:
            if not last_round:
                return None, self.fallback_json_request(node_id)
            else:
                return None, self.fallback_answer_request(node_id)

        return messages_sent, None

    async def update_messages(self, results):
        fallback_tasks = []
        fallback_nodes = []
        for node, result in zip(self.graph.nodes(), results):
            message, fallback = result
            if message is None:
                fallback_nodes.append(node)
                fallback_tasks.append(fallback)
            else:
                self.messages[node] = message

        if len(fallback_tasks) > 0:
            messages = await asyncio.gather(*fallback_tasks)
            for node, message in zip(fallback_nodes, messages):
                self.messages[node] = message

    async def bootstrap(self):
        """Bootstrap all nodes asynchronously."""
        tasks = [self.bootstrap_node(v) for v in self.graph.nodes()]
        results = await asyncio.gather(*tasks)
        await self.update_messages(results)

    async def bootstrap_node(self, node_id):
        """Bootstraps node with task-specific instructions."""
        bootstrap_parameters = self.get_bootstrap_parameters(node_id)

        system_message = SystemMessage(content=self.bootstrap_template.format(**bootstrap_parameters))
        user_message = HumanMessage(content=self.bootstrap_ask_for_first_messages.format(cot_prompt= self.cot_prompt))

        config = {"configurable": {"thread_id": str(node_id)}}
        response = await self.app.ainvoke({"messages": [system_message, user_message]}, config=config)
        self.chat_history[node_id] = messages_to_dict(response['messages'])

        return await self.parse_response_to_dict(self.chat_history[node_id], node_id)
    
    def get_bootstrap_parameters(self, node_id):
        neighbors = ", ".join(str(self.graph.nodes[i]["name"]) for i in self.graph.neighbors(node_id))

        return {
            "short_task_description": self.short_task_description,
            "long_problem_description": self.long_problem_description,
            "num_agents": self.graph.order(),
            "own_name": self.graph.nodes[node_id]["name"],
            "neighbors": neighbors,
            "num_rounds": self.rounds,
        }

    async def message_passing(self, node_id: int, rounds_left: int, messages: dict[str, str]):
        """Handles message exchange between nodes."""
        last_round = rounds_left == 0
        if last_round:
            messages_str = "Message passing has finished, here are the last messages you got from your neighbors:\n\n"
        else:
            messages_str = "These are the messages from your neighbors:\n\n"
        for name, message in messages.items():
            messages_str += f"Message from {name}:\n\n{message}\n\n"

        neighbour_names = [str(self.graph.nodes[i]["name"]) for i in self.graph.neighbors(node_id) if i not in messages]
        silent_neighbors = [name for name in neighbour_names if name not in messages]
        if len(silent_neighbors) > 0:
            messages_str += f"The following neighbors did not send you a messages in this round: {', '.join(silent_neighbors)}\n\n"

        if not last_round:
            neighbors = ", ".join(neighbour_names)
            messages_str += f"{self.cot_prompt} Output your messages in JSON format as specified earlier. You have {rounds_left} rounds of communication left before you need to decide. Your neighbors are: {neighbors} "
            if rounds_left == 1:
                messages_str += "These are the last messages that your neighbors will receive from you."
        else:
            messages_str += self.cot_prompt_final_prediction + self.format_instructions.format(question=self.question_for_prediction, valid_answers=", ".join(self.get_valid_answers(node_id)))

        user_message = HumanMessage(content=messages_str)

        config = {"configurable": {"thread_id": str(node_id)}}
        response = await self.app.ainvoke({'messages': [user_message]}, config=config)
        self.chat_history[node_id] = messages_to_dict(response['messages'])

        return await self.parse_response_to_dict(self.chat_history[node_id], node_id, last_round)

    async def pass_messages(self):
        """Executes synchronous message passing rounds."""
        for round in range(1, self.rounds+1):
            rounds_left = self.rounds - round
            all_messages_sent = {}
            for v in self.graph.nodes():
                all_messages_sent[v] = self.messages[v]

            tasks = []
            for v in self.graph.nodes():
                name = self.graph.nodes[v]["name"]
                messages_received = {}
                for neigh in self.graph.neighbors(v):
                    if name in all_messages_sent[neigh]:
                        neigh_name = self.graph.nodes[neigh]["name"]
                        messages_received[neigh_name] = all_messages_sent[neigh][name]

                tasks.append(self.message_passing(node_id=v, rounds_left=rounds_left, messages=messages_received))

            results = await asyncio.gather(*tasks)
            await self.update_messages(results)

        return self.get_final_answers(self.messages, len(self.graph.nodes))
    
    def get_final_answers(self, last_messages, num_nodes):
        answers = [None] * num_nodes
        for node in last_messages.keys():
            answers[node] = last_messages[node]
        return answers

    def get_transcripts(self):
        """Returns the full transcript of all messages exchanged."""
        return {self.graph.nodes[v]['name']: self.chat_history[v] for v in self.graph.nodes()}

    def get_score(self, answers: list[str]) -> float:
        raise NotImplementedError
    
    def get_valid_answers(self, node_id):
        raise NotImplementedError


class Consensus(LiteralMessagePassing):
    """Implementation for consensus without malicious parties."""
    def __init__(self, graph, rounds, model_name="gpt-4o", model_provider="openai", chain_of_thought=True):
        self.rounds = rounds
        self.short_task_description = "Your goal is to agree on a single value with the other agents. The possible values that you can decide on are either 0 or 1."
        self.long_problem_description = (
            "After the last round, each agent must decide on a single value."
        )
        self.question_for_prediction = "Please answer with the value you agreed on."

        super().__init__(graph, model_name, model_provider, chain_of_thought)

    def get_score(self, answers: list[str]) -> float:
        valid_values = {"0", "1"}
        if not all(x in valid_values for x in answers) or (
                answers.count('0') != self.graph.order() and answers.count('0') != 0):
            return 0.0
        else:
            return 1.0
        
    def get_valid_answers(self, node_id):
        return ["0", "1"]


class LeaderElection(LiteralMessagePassing):
    def __init__(self, graph, rounds, model_name="gpt-4o", model_provider="openai", chain_of_thought=True):
        self.rounds = rounds
        self.short_task_description = "Your task is to collaboratively solve the problem of electing a single leader."
        self.long_problem_description = (
            "You will be requested to state whether or not you are the leader. The response should either be 'Yes' or 'No'. "
            "The final result should be such that exactly one agent responds with 'Yes' and all others say 'No' as there should be exactly one leader."
        )
        self.question_for_prediction = "Are you the leader?"
        super().__init__(graph, model_name, model_provider, chain_of_thought)

    def get_score(self, answers: list[str]) -> float:
        valid_values = {"No", "Yes"}
        all_valid = all(x in valid_values for x in answers)
        one_leader = len([x for x in answers if x == "Yes"]) == 1
        return float(all_valid and one_leader)
    
    def get_valid_answers(self, node_id):
        return ["Yes", "No"]


class Matching(LiteralMessagePassing):
    def __init__(self, graph, rounds, model_name="gpt-4o-mini", model_provider="openai", chain_of_thought=True):
        self.rounds = rounds
        self.short_task_description = "Your task is to find build groups of two agents each which can communicate with each other."
        self.long_problem_description = (
            "You will be requested to name one of your neighbors that you build a group with or 'None' if all your neighbors are already assigned to other groups and cannot be in a group with you." \
            "In the end, every agent should only be in at most one group and agents in the same group have to name each other as the second group member consistently."
        )
        self.question_for_prediction = "Please answer with the name of the neighbor you build a group with or 'None' if all your neighbors are already assigned to other groups."
        super().__init__(graph, model_name, model_provider, chain_of_thought)

    def get_score(self, answers: list[str]) -> float:
        graph = self.graph
        node_names = [graph.nodes[node]['name'] for node in graph.nodes]

        name_to_match = {node_names[i]: answers[i] for i in range(len(node_names))}

        inconsistent_count = 0
        for node in graph.nodes:
            matching_node = answers[node]
            if matching_node != 'None':
                if matching_node not in [node_names[u] for u in graph.neighbors(node)]:
                    inconsistent_count += 1
                elif name_to_match[matching_node] != node_names[node]:
                    inconsistent_count += 1
            else:
                for v in graph.neighbors(node):
                    if answers[v] == 'None':
                        inconsistent_count += 1
                        break

        return (graph.order() - inconsistent_count) / graph.order()
    
    def get_valid_answers(self, node_id):
        return [self.graph.nodes[neigh]["name"] for neigh in self.graph.neighbors(node_id)] + ["None"]


class Coloring(LiteralMessagePassing):
    def __init__(self, graph: nx.Graph, rounds: int, num_colors: int | None = None, model_name="gpt-4o-mini", model_provider="openai", chain_of_thought=True):
        super().__init__(graph, model_name, model_provider, chain_of_thought)
        self.rounds = rounds

        if num_colors is not None:
            self.num_colors = num_colors
        else:
            self.num_colors = max(graph.degree(n) for n in graph.nodes) + 1

        self.colors = [f"Group {i+1}" for i in range(self.num_colors)]

        self.short_task_description = "Your task is to partition yourselves into groups such that agents who are neighbors are never in the same group."
        self.long_problem_description = (
            f"You will be requested to state which group you assign yourself to. There are exactly {self.num_colors} groups available: Group 1,...,Group {self.num_colors}. You should assign yourself to exactly one of these groups. "
            "The final result should be such that any two agents who are neighbors are in different groups. In particular, you should assign yourself to a group that is different from all of your neighbors' groups. "
        )
        self.question_for_prediction = f"Which group do you assign yourself to?"

    def get_score(self, answers: list[str]) -> float:
        all_valid = all(x in self.colors for x in answers)
        valid_edges = [answers[u] != answers[v] for (u, v) in self.graph.edges].count(True)
        valid_ratio = valid_edges / self.graph.number_of_edges()
        return float(all_valid) * valid_ratio
    
    def get_valid_answers(self, node_id):
        return self.colors


def score_vertex_cover(results, graph):
    def vertex_cover(results):
        covered_edges = sum([1 for u, v in graph.edges if (results[u] == "Yes" or results[v] == "Yes")])
        return covered_edges / graph.number_of_edges()

    coverage = vertex_cover(results)

    minimality = 0
    cover_size = 0
    for u in graph.nodes:
        if results[u] is not None:
            if results[u] == "Yes":
                cover_size += 1
                _results = results.copy()
                _results[u] = "No"
                if vertex_cover(_results) < 1.0:
                    minimality += 1

    return coverage * minimality / cover_size


class VertexCover(LiteralMessagePassing):
    def __init__(self, graph, rounds, model_name="gpt-4o-mini", model_provider="openai", chain_of_thought=True):
        """https://math.stackexchange.com/a/1764484.
        A practical example is that the minimal vertex cover receives resources and is 
        important that every channel of communication always has access to this resource,
        meaning there is no need for two-hop communication to obtain some resource. Fundamentally,
        the agents solve a resource allocation problem, which also touches into fairness.
        """
        super().__init__(graph, model_name, model_provider, chain_of_thought)
        self.rounds = rounds
        self.short_task_description = """Your task is to select, among all agents, a group of coordinators
such that whenever two agents communicate at least one of them is a coordinator. The group of coordinators
should be selected such that every coordinator has at least one neighbor who is not a coordinator.
"""
        # NOTE: Closer to definition
        # reverting any coordinator back to a regular agent results in at least two agents 
        # who can communicate but none of whom is a coordinator.

        self.long_problem_description = (
            "You will be requested to state whether you are a coordinator. The response should either be 'Yes' or 'No'. "
        )
        self.question_for_prediction = "Are you a coordinator?"

    def is_vertex_cover(self, results):
        covered_edges = sum([1 for u, v in self.graph.edges if (results[u] == "Yes" or results[v] == "Yes")])
        return covered_edges == self.graph.number_of_edges()

    def get_score(self, results: list[str]) -> float:
        return score_vertex_cover(results, self.graph)
    
    def get_valid_answers(self, node_id):
        return ["Yes", "No"]
