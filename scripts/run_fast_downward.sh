#!/bin/bash

# This script run fast downward on a folder of PDDL problems
# Assumes that fast downward is located in the same folder as the script
# Usage: `bash run_fast_downward.sh <path_to_pddl_folder> <only_ground> <output_folder> <sas_file> <plan_file> <remove_output_folder> <time_limit> <memory_limit> <verboose>`
# Example: `bash run_fast_downward.sh data/pddl/quantified 0`
# -- only_ground: 1 if you want to skip quantified problems, 2 if you want to skip grounded problems, 0 otherwise
# -- remove_output_folder: 1 if the output folder should be deleted, 0 otherwise. Default is 0
# -- time_limit: time limit for fast downward. Default is 30m
# -- memory_limit: memory limit for fast downward. Default is 0 e.g. no memory limit
# -- verboose: 1 for verboose output, 0 otherwise. Default is 1

# Heuristics to test:
# - --alias lama-first, fast but satisficing
# - --search "astar(blind())", optimal but slow
# - --search "astar(hmax())", optimal but slow
# - --search "astar(ff())", faster but not optimal
# - --search "astar(merge_and_shrink())", quantified variables should be supported (https://www.fast-downward.org/Doc/Evaluator#Merge-and-shrink_heuristic)
# but does not seem to work with the setup below:
# "merge_and_shrink(shrink_strategy=shrink_bisimulation(greedy=false),merge_strategy=merge_sccs(order_of_sccs=topological,merge_selector=score_based_filtering(scoring_functions=[goal_relevance(),dfp(),total_order()])),label_reduction=exact(before_shrinking=true,before_merging=false),max_states=50k,threshold_before_merge=1)"

# set -e

pddl_folder=$1
only_ground=$2
output_folder=$3
sas_file=$4
plan_file=$5
rm_out_folder=$6
time_limit=$7
memory_limit=$8
verboose=$9
# output_folder="fd_out"

if [ -z "$only_ground" ]; then
  echo "No 'only_ground' argument provided. Defaulting to false"
  only_ground=0
fi

if [ -z "$output_folder" ]; then
  echo "No 'output_folder' argument provided. Defaulting to 'fd_out'"
  output_folder="fd_out"
fi

if [ -z "$sas_file" ]; then
  echo "No 'sas_file' argument provided. Defaulting to 'output.sas'"
  sas_file="output.sas"
fi

if [ -z "$plan_file" ]; then
  echo "No 'plan_file' argument provided. Defaulting to 'sas_plan'"
  plan_file="sas_plan"
fi

if [ -z "$rm_out_folder" ]; then
  echo "No 'rm_out_folder' argument provided. Defaulting to false"
  rm_out_folder=0
fi

if [ -z "$time_limit" ]; then
  echo "No 'time_limit' argument provided. Defaulting to 30m"
  time_limit="30m"
fi

if [ -z "$memory_limit" ]; then
  echo "No 'memory_limit' argument provided. Defaulting to False"
  memory_limit=0
fi

if [ -z "$verboose" ]; then
  echo "No 'verboose' argument provided. Defaulting to True"
  verboose=1
fi

if [ $rm_out_folder == 1 ]; then
  echo "Removing output folder"
  rm -rf $output_folder
fi
mkdir -p $output_folder

for file in $pddl_folder/*; do
  if [[ "$file" == *"domain"* ]]; then
    if [ $verboose == 1 ]; then
      echo "Skipping domain file"
    fi
  elif [ $only_ground == 1 ]; then
    if [[ "$file" == *"ground"* ]]; then
      if [ $verboose == 1 ]; then
        echo "Running fast downward on $file"
      fi
      if [ $memory_limit == 1 ]; then
        ./fast-downward.sif --alias lama-first --overall-time-limit $time_limit --overall-memory-limit 8g --sas-file $sas_file --plan-file $plan_file $file > $output_folder/$(basename $file).txt
      else
        ./fast-downward.sif --alias lama-first --overall-time-limit $time_limit --sas-file $sas_file --plan-file $plan_file $file > $output_folder/$(basename $file).txt
      fi
    fi
  elif [ $only_ground == 2 ]; then
    if [[ "$file" == *"quant"* ]]; then
      if [ $verboose == 1 ]; then
        echo "Running fast downward on $file"
      fi
      if [ $memory_limit == 1 ]; then
        ./fast-downward.sif --alias lama-first --overall-time-limit $time_limit --overall-memory-limit 8g --sas-file $sas_file --plan-file $plan_file $file > $output_folder/$(basename $file).txt
      else
        ./fast-downward.sif --alias lama-first --overall-time-limit $time_limit --sas-file $sas_file --plan-file $plan_file $file > $output_folder/$(basename $file).txt
      fi
    fi
  else
    if [ $verboose == 1 ]; then
      echo "Running fast downward on $file"
    fi
    if [ $memory_limit == 1 ]; then
      ./fast-downward.sif --alias lama-first --overall-time-limit $time_limit --overall-memory-limit 8g --sas-file $sas_file --plan-file $plan_file $file > $output_folder/$(basename $file).txt
    else
      ./fast-downward.sif --alias lama-first --overall-time-limit $time_limit --sas-file $sas_file --plan-file $plan_file $file > $output_folder/$(basename $file).txt
    fi
  fi
done