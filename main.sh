#!/bin/bash

for graph_size in 4; do
    for task in coloring leader_election matching vertex_cover consensus; do
        for rounds in 4; do
            python main.py --graph_size $graph_size --task $task --rounds $rounds --samples_per_graph_model 3 --seed 1729 --model $1 --start_from_sample $2
done; done; done;

for graph_size in 8; do
    for task in coloring leader_election matching vertex_cover consensus; do
        for rounds in 5; do
            python main.py --graph_size $graph_size --task $task --rounds $rounds --samples_per_graph_model 3 --seed 1729 --model $1 --start_from_sample $2
done; done; done;

for graph_size in 16; do
    for task in coloring leader_election matching vertex_cover consensus; do
        for rounds in 6; do
            python main.py --graph_size $graph_size --task $task --rounds $rounds --samples_per_graph_model 3 --seed 1729 --model $1 --start_from_sample $2
done; done; done;


