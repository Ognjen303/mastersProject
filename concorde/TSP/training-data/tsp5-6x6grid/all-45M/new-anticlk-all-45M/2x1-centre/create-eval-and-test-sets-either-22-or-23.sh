#!/bin/bash

# File containing shuffled examples
input_file="shuffled-either-22-or-23-present.txt"

# Get the number of lines in the input file
total_lines=$(wc -l < "$input_file")

# Output files
eval_file="eval-either-22-or-23.txt"
test_file="test-either-22-or-23.txt"

# Number of lines in eval set
eval_lines=27000

# Start and end lines for test set
# NOTE: We included the first 1000000 lines in the train-1M-either-22-or-23.txt train set
# That is why we start the test set from line 1000001
test_start_line=1000001 
test_end_line=$((total_lines - eval_lines))

# Create eval set
tail -n $eval_lines $input_file > $eval_file

# Create test set
sed -n "${test_start_line},${test_end_line}p" $input_file > $test_file

echo "Eval set ($eval_lines lines) created: $eval_file"
echo "Test set ($((test_end_line - test_start_line + 1)) lines) created: $test_file"
