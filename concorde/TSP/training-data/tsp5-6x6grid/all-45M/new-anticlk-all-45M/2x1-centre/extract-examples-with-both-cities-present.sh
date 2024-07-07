#!/bin/bash

# Define the input file
input_file="shuffled-2x1centre.txt"

# Define the output file
output_file1="cities-22-23.txt"
output_file2="shuffled-either-22-or-23-present.txt"

# Extract lines containing 2 2 and 2 3 in any order
grep -E "((2 2.*2 3)|(2 3.*2 2))" "$input_file" > "$output_file1"

# Extract lines that don't contain both 2 2 and 2 3 at the same time
grep -v -E "((2 2.*2 3)|(2 3.*2 2))" "$input_file" > "$output_file2"

echo "Extraction completed."