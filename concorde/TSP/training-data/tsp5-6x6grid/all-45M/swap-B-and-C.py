import argparse
import numpy as np
import pdb

# Parse the command-line arguments
parser = argparse.ArgumentParser()

# Add mandatory arguments
parser.add_argument('-i', '--input_file', help='Input dataset', default='solutions-1k-tsp5.txt')
parser.add_argument('-o', '--output_file', help='Output file', default='swapped-solutions-1k-tsp5.txt')
parser.add_argument('-c', '--count')

# Parse the command-line arguments
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file
count = int(args.count)


# Function to swap two characters in a string
def swap_characters(my_string):
    # Remove spaces from the string and split into a list
    lst = my_string.strip()
    lst = lst.replace(' ', '')
    lst = list(lst)
    index_b = lst.index('B')
    index_c = lst.index('C')
    i, j = sorted((index_b, index_c))
    # Perform the swap
    lst[i], lst[j] = lst[j], lst[i]
    return ' ' + ' '.join(lst) + '\n'

# Read the original file
with open(input_file, 'r') as file:
    lines = file.readlines()


def nearly_equidistant_numbers(count, upper=100):
    """
    Generates a set of nearly equidistant numbers between 0 and 100.

    Parameters:
        count (int): The number of nearly equidistant numbers to generate.

    Returns:
        set: A set of nearly equidistant numbers between 0 and 100.
    """

    # Calculate the step size
    step_size = upper / (count - 1)
    
    # Initialize the result set
    result = set()
    
    # Generate the nearly equidistant numbers
    for i in range(count):
        next_number = round(step_size * i)
        result.add(next_number)
    
    return sorted(result)



# Open a new file for writing swapped lines
with open(output_file, 'w') as file:

    upper = 100
    remainders = nearly_equidistant_numbers(count, upper=upper)

    #remainders = range(100)
    

    for i, line in enumerate(lines, start=1):
        # Randomly decide whether to swap characters
        if (i % upper) in remainders:
            # Split the line into two parts before and after the semicolon
            parts = line.split(';')
            sequence = parts[1]  # Take characters after semicolon for swapping
            swapped_sequence = swap_characters(sequence) # swap cities B and C
            # Write the swapped line to the new file
            file.write(parts[0]+ ';' + swapped_sequence)
        else:
            # If no swap, just write the original line
            file.write(line)
