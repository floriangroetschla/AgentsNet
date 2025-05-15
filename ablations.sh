#!/bin/bash

for graph_size in 20 30 40 50 60 70 80 90 100; do
    for task in coloring leader_election matching vertex_cover consensus; do
        python main.py --graph_size $graph_size --task $task --samples_per_graph_model 3 --seed 1729 --model $1
done; done;


