#!/bin/bash

# Define the input file
input_file="shuffled-centre.txt"

# Define the output file
output_file1="cities-22-23-32.txt"
output_file2="cities-22-23-33.txt"
output_file3="cities-22-32-33.txt"
output_file4="cities-23-32-33.txt"

# Extract lines containing 2 2 and 2 3 in any order, but do not contain 3 2 or 3 3
grep -E "(2 2.*2 3.*3 2|2 2.*3 2.*2 3|2 3.*2 2.*3 2|2 3.*3 2.*2 2|3 2.*2 2.*2 3|3 2.*2 3.*2 2)" "$input_file" | grep -Ev "3 3" > "$output_file1"

# Extract lines containing all of 2 2, 2 3, and 3 3 in any order, but do not contain 3 2
grep -E "(2 2.*2 3.*3 3|2 2.*3 3.*2 3|2 3.*2 2.*3 3|2 3.*3 3.*2 2|3 3.*2 2.*2 3|3 3.*2 3.*2 2)" "$input_file" | grep -Ev "3 2" > "$output_file2"

# Extract lines containing all of 2 2, 3 2, and 3 3 in any order, but do not contain 2 3
grep -E "(2 2.*3 2.*3 3|2 2.*3 3.*3 2|3 2.*2 2.*3 3|3 2.*3 3.*2 2|3 3.*2 2.*3 2|3 3.*3 2.*2 2)" "$input_file" | grep -Ev "2 3" > "$output_file3"

# Extract lines containing all of 2 3, 3 2, and 3 3 in any order, but do not contain 2 2
grep -E "(2 3.*3 2.*3 3|2 3.*3 3.*3 2|3 2.*2 3.*3 3|3 2.*3 3.*2 3|3 3.*2 3.*3 2|3 3.*3 2.*2 3)" "$input_file" | grep -Ev "2 2" > "$output_file4"


echo "Extraction completed."