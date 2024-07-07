#!/bin/bash

# Define the input file
input_file="shuffled.txt"

# Define the output files
train_no_centre_file="shuffled-no-centre.txt"
centre_file="shuffled-centre.txt"

# Define the patterns to search for
patterns="2 2|2 3|3 2|3 3"

# Use grep to extract lines containing any of the specified patterns to centre_file
grep -E "$patterns" "$input_file" > "$centre_file"

# Use grep to extract lines not containing any of the specified patterns to train_no_centre_file
grep -v -E "$patterns" "$input_file" > "$train_no_centre_file"

echo "Extraction completed. Results can be found in $train_no_centre_file and $centre_file."
