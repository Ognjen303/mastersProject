import argparse
import subprocess
import pdb
from math import floor


# Merge files into output file
def merge_files(file1_path, file2_path, output_path):
    """
    Merge the contents of two files, with lines from file2 interspersed among lines from file1,
    such that the lines from file2 are equidistant from each other inside the output file.

    Args:
        file1_path (str): Path to the first input file.
        file2_path (str): Path to the second input file.
        output_path (str): Path to the output file.

    Returns:
        int: The number of lines written to the output file.
    """

    # Count lines in file1
    lines_file1 = int(subprocess.check_output(['wc', '-l', file1_path]).split()[0])
    # Count lines in file2
    lines_file2 = int(subprocess.check_output(['wc', '-l', file2_path]).split()[0])

    assert lines_file1 > lines_file2, "First file must be larger that second file."

    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2, open(output_path, 'w') as out:
        
        lines_written = 0

        q = int(floor(lines_file1 / lines_file2)) # quotionent
        r_orig = lines_file1 - q*lines_file2 # remainer

        # initialise
        r = r_orig
        r_dash = r_orig
        c = 0
        total_caries = 0

        while lines_written < lines_file1:
            
            
            c = int(floor(r / lines_file2))

            chunk_size = min(lines_file1 - lines_written, q+c)


            out.writelines(f1.readline() for _ in range(chunk_size))
            if lines_written + chunk_size <= lines_file1:
                out.write(f2.readline())
            lines_written += chunk_size

            # Reset counter
            if c > 0:
                total_caries += c
                c = 0
                r_dash = r - lines_file2 # remove the 'integer' part
                r = r_dash
            
            # add remainer
            r += r_orig
            

    print(f'{total_caries=}')
    return lines_written

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Merge two files.')
parser.add_argument('file1', help='Path to the first file')
parser.add_argument('file2', help='Path to the second file')
parser.add_argument('output', help='Path to the output file')

args = parser.parse_args()

# Merge files
merge_files(args.file1, args.file2, args.output)
print(f"Merge completed! Output can be found in {args.output}")

# Verify if the output file has as many lines as the sum of lines from file1 and file2
expected_lines = int(subprocess.check_output(['wc', '-l', args.file1]).split()[0]) + int(subprocess.check_output(['wc', '-l', args.file2]).split()[0])
actual_lines = int(subprocess.check_output(['wc', '-l', args.output]).split()[0])
assert actual_lines == expected_lines, "Output file does not have the expected number of lines"
print("Assertion passed: Output file has the expected number of lines")
