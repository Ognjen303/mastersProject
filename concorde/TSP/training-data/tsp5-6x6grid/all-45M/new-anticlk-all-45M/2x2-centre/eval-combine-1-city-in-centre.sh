#!/bin/bash

# Define the input files
input_file_2_2="lines_with_2_2.txt"
input_file_2_3="lines_with_2_3.txt"
input_file_3_2="lines_with_3_2.txt"
input_file_3_3="lines_with_3_3.txt"

# Define the output file
one_city_eval_file="eval-one-city.txt"

# Extract 6750 random lines from each input file and concatenate them into one file
shuf -n 6750 "$input_file_2_2" >> "$one_city_eval_file"
shuf -n 6750 "$input_file_2_3" >> "$one_city_eval_file"
shuf -n 6750 "$input_file_3_2" >> "$one_city_eval_file"
shuf -n 6750 "$input_file_3_3" >> "$one_city_eval_file"

echo "Extraction completed."
