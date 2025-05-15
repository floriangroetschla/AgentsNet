import textwrap
import json
import argparse
from itertools import zip_longest

from LiteralMessagePassing import parse_messages

COLORS = {
    "human": "\033[96m", # cyan
    "ai": "\033[92m", # green
    "system": "\033[93m", # yellow
    "reset": "\033[0m",
    "agent1": "\033[92m",  # green
    "agent2": "\033[93m",  # yellow
}

def format_message(role, message, width=60, indent=12, name=None):
    wrapped_text = textwrap.fill(message, width)
    indented_text = textwrap.indent(wrapped_text, " " * indent)
    name = role if name is None else name
    name = COLORS[role] + name + COLORS["reset"]
    return f"{name}:\n{indented_text}\n"

def wrap_preserve_newlines(text, width):
    lines = text.splitlines()
    wrapped_lines = []
    for line in lines:
        # Keep empty lines intact
        if line.strip() == '':
            wrapped_lines.append('')
        else:
            wrapped_lines.extend(textwrap.wrap(line, width=width))
    return wrapped_lines

def print_side_by_side(text1, text2, width=80, padding=4):
    wrapped1 = wrap_preserve_newlines(text1, width)
    wrapped2 = wrap_preserve_newlines(text2, width)

    for line1, line2 in zip_longest(wrapped1, wrapped2, fillvalue=''):
        print(f"{line1:<{width}}{' ' * padding}{line2:<{width}}")

def collect_messages(chat, agent_name):
    responses = chat[agent_name]
    msg_list = []
    for response_dict in responses:
        msg_dict = parse_messages(response_dict["data"]["content"])
        if msg_dict is not None:
            msg_list.append(msg_dict)
    return msg_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--agents", type=str, nargs="+", default=[])
    args = parser.parse_args()

    assert len(args.agents) < 3

    with open(args.file, "r") as fp:
        chat = json.load(fp)["transcripts"]

    if len(args.agents) < 2:
        for name, messages in chat.items():
            if len(args.agents) == 0 or args.agents[0] == name:
                print(f"========\nChat history of {name}\n========\n")
                for message_dict in messages:
                    message = message_dict["data"]["content"]
                    role = message_dict["type"]
                    print(format_message(role, message))
    else:
        agent1, agent2 = args.agents
        msg_dicts_1 = collect_messages(chat, agent1)
        msg_dicts_2 = collect_messages(chat, agent2)
        round = 1
        for msg_dict_1, msg_dict_2 in zip(msg_dicts_1, msg_dicts_2):
            print(f"{'=' * 74} Round {round} {'=' * 74}")
            msg1 = msg_dict_1[agent2] if agent2 in msg_dict_1 else "### No Message ###"
            msg2 = msg_dict_2[agent1] if agent1 in msg_dict_2 else "### No Message ###"
            msg1 = format_message("agent1", msg1, name=agent1)
            msg2 = format_message("agent2", msg2, name=agent2)
            print_side_by_side(msg1, msg2)
            round += 1
