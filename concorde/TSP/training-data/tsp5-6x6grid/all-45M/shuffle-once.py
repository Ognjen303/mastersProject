from sklearn.utils import shuffle
import os

# Set the seed for reproducibility
random_state = 42

# Define input and output file paths
input_file = 'remaining-45M.txt'
output_file = 'shuffle-remaining-45M.txt'

# Read lines from the input file
with open(input_file, 'r') as f:
    lines = f.readlines()

# Shuffle the lines
shuffled_lines = shuffle(lines, random_state=random_state)

# Write shuffled lines to the output file
with open(output_file, 'w') as f:
    f.writelines(shuffled_lines)

print("Shuffling complete. Output written to:", output_file)
