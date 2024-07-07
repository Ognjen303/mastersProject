import random
import argparse
import numpy as np


def random_permutation(input_str):
    # Split the input string into a list of letters

    letters = input_str.strip().split()
    original_seq = ' '.join(letters)
    shuffled_seq = original_seq

    while original_seq == shuffled_seq:
        random.shuffle(letters)
        shuffled_seq = ' '.join(letters)
    
    return shuffled_seq + '\n'



# Parse the command-line arguments
parser = argparse.ArgumentParser()

# Add mandatory arguments
parser.add_argument('-i', '--input_file', help='Input dataset', default='solutions-1k-tsp5.txt')
parser.add_argument('-o', '--output_file', help='Output file', default='swapped-solutions-1k-tsp5.txt')

# Parse the command-line arguments
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file


# Read the original file
with open(input_file, 'r') as file:
    lines = file.readlines()


# Open a new file for writing swapped lines
with open(output_file, 'w') as file:
    for line in lines:
        # Split the line into two parts before and after the semicolon
        parts = line.split('; A ')
        sequence = parts[1]  # Take characters after semicolon for swapping
        random_seq = random_permutation(sequence)
        # Write the swapped line to the new file
        file.write(parts[0]+ '; A ' + random_seq)
