#!/bin/bash

# Define the input files
input_file1="cities-22-23.txt"
input_file2="cities-22-32.txt"
input_file3="cities-22-33.txt"
input_file4="cities-23-32.txt"
input_file5="cities-23-33.txt"
input_file6="cities-32-33.txt"

# Define the output file
two_city_eval_file="eval-two-city.txt"

# Extract 4500 random lines from each input file and concatenate them into one file
shuf -n 4500 "$input_file1" >> "$two_city_eval_file"
shuf -n 4500 "$input_file2" >> "$two_city_eval_file"
shuf -n 4500 "$input_file3" >> "$two_city_eval_file"
shuf -n 4500 "$input_file4" >> "$two_city_eval_file"
shuf -n 4500 "$input_file5" >> "$two_city_eval_file"
shuf -n 4500 "$input_file6" >> "$two_city_eval_file"

echo "Extraction completed."
