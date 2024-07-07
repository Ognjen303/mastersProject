#!/bin/bash

# Define the input file
input_file="shuffled-centre.txt"

# Define the output files
pattern_2_2_file="lines_with_2_2.txt"
pattern_2_3_file="lines_with_2_3.txt"
pattern_3_2_file="lines_with_3_2.txt"
pattern_3_3_file="lines_with_3_3.txt"

# Extract lines containing 2 2 but not 2 3, 3 2, or 3 3
grep -E "2 2" "$input_file" | grep -Ev "(2 3|3 2|3 3)" > "$pattern_2_2_file"

# Extract lines containing 2 3 but not 2 2, 3 2, or 3 3
grep -E "2 3" "$input_file" | grep -Ev "(2 2|3 2|3 3)" > "$pattern_2_3_file"

# Extract lines containing 3 2 but not 2 2, 2 3, or 3 3
grep -E "3 2" "$input_file" | grep -Ev "(2 2|2 3|3 3)" > "$pattern_3_2_file"

# Extract lines containing 3 3 but not 2 2, 2 3, or 3 2
grep -E "3 3" "$input_file" | grep -Ev "(2 2|2 3|3 2)" > "$pattern_3_3_file"

echo "Extraction completed."
