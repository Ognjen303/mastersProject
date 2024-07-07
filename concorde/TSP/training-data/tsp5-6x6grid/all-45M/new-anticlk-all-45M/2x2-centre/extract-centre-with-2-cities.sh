#!/bin/bash

# Define the input file
input_file="shuffled-centre.txt"

# Define the output file
output_file1="cities-22-23.txt"
output_file2="cities-22-32.txt"
output_file3="cities-22-33.txt"
output_file4="cities-23-32.txt"
output_file5="cities-23-33.txt"
output_file6="cities-32-33.txt"

# Extract lines containing 2 2 and 2 3 in any order, but do not contain 3 2 or 3 3
grep -E "((2 2.*2 3)|(2 3.*2 2))" "$input_file" | grep -Ev "(3 2|3 3)" > "$output_file1"
grep -E "((2 2.*3 2)|(3 2.*2 2))" "$input_file" | grep -Ev "(2 3|3 3)" > "$output_file2"
grep -E "((2 2.*3 3)|(3 3.*2 2))" "$input_file" | grep -Ev "(2 3|3 2)" > "$output_file3"
grep -E "((2 3.*3 2)|(3 2.*2 3))" "$input_file" | grep -Ev "(2 2|3 3)" > "$output_file4"
grep -E "((2 3.*3 3)|(3 3.*2 3))" "$input_file" | grep -Ev "(2 2|3 2)" > "$output_file5"
grep -E "((3 2.*3 3)|(3 3.*3 2))" "$input_file" | grep -Ev "(2 2|2 3)" > "$output_file6"

echo "Extraction completed."