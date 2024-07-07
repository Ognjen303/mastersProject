#!/bin/bash

# Define the input files
input_file1="cities-22-23-32.txt"
input_file2="cities-22-23-33.txt"
input_file3="cities-22-32-33.txt"
input_file4="cities-23-32-33.txt"

# Define the output file
three_city_eval_file="eval-three-city.txt"

# Extract 6750 random lines from each input file and concatenate them into one file
shuf -n 6750 "$input_file1" >> "$three_city_eval_file"
shuf -n 6750 "$input_file2" >> "$three_city_eval_file"
shuf -n 6750 "$input_file3" >> "$three_city_eval_file"
shuf -n 6750 "$input_file4" >> "$three_city_eval_file"

echo "Extraction completed."
