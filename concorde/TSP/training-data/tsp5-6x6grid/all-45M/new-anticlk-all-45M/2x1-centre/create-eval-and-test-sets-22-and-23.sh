#!/bin/bash

# File containing shuffled examples
input_file="cities-22-23.txt"

# Get the number of lines in the input file
total_lines=$(wc -l < "$input_file")

# Output files
eval_file="eval-22-and-23-at-same-time.txt"
test_file="test-22-and-23-at-same-time.txt"

# Number of lines in eval set
eval_lines=27000

test_start_line=1
test_end_line=$((total_lines - eval_lines))

# Create eval set
tail -n $eval_lines $input_file > $eval_file

# Create test set
sed -n "${test_start_line},${test_end_line}p" $input_file > $test_file

echo "Eval set ($eval_lines lines) created: $eval_file"
echo "Test set ($((test_end_line - test_start_line + 1)) lines) created: $test_file"
