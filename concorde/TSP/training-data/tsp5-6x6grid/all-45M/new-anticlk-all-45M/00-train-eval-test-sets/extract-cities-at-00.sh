#!/bin/bash

# Define the input file
input_file="shuffled.txt"

# Define the output files
train_no_00="train-no-00.txt"
file_00="shuffled-00-only.txt"
eval_00="eval-00.txt"

# Define the patterns to search for
patterns="0 0"

# Use grep to extract lines containing any of the specified pattern
grep -E "$patterns" "$input_file" > "$file_00"

# Use grep to extract lines not containing any of the specified patterns
grep -v -E "$patterns" "$input_file" > "$train_no_00"

# Eval data consisting of only examples at coordinate 0 0
shuf -n 27000 "$file_00" >> "$eval_00"

echo "Extraction completed. Results can be found in $train_no_00, $file_00 and $eval_00."
