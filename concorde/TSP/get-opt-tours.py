import subprocess
import os
import sys
import re
import pdb
import time
import glob
import argparse


def add_city_letters(line):
    """
    Given a set of x & y coordinates of cities in TSP, we want to denote each city
    with a letter. Best to explain on an example. Given an input of 10 cities:
    
    0 4 3 5 6 5 7 0 1 1 3 1 4 7 5 8 0 2 8 7                                (1)

    Where
    0 4 are the x & y coordinates of zeroth city
    3 5 are the x & y coordinates of first city etc.

    We want to return a sequence:

    A 0 4 B 3 5 C 6 5 D 7 0 E 1 1 F 3 1 G 4 7 H 5 8 I 0 2 J 8 7            (2)

    So now the zeroth city is denoted with A followed by its x & y coordinates,
    first city is B etc.

    Additionally, it returns a dictionary mapping city numbers to letters.

    args:
    line (string) -> a TSP setup sequence like (1)
    output:
    new_line (string) -> a sequence like (2)
    city_mapping (dict) -> a dictionary mapping city numbers to letters
    """

    coordinates = line.split()
    num_of_coordinates = len(coordinates)
    
    if num_of_coordinates % 2 != 0:
        raise ValueError("Sequence of city coordinates must have an even number of entries.")
    
    new_line = ""
    city_mapping = {}

    for i in range(0, num_of_coordinates, 2):
        city_letter = chr(ord('A') + i // 2)  # Convert index to corresponding letter
        city_mapping[str(i // 2)] = city_letter
        new_line += f"{city_letter} {coordinates[i]} {coordinates[i + 1]} "
    
    assert len(city_mapping) == (num_of_coordinates // 2)
    
    return new_line, city_mapping




def write_tsplib_file(file_path, coordinates):

    """Convert a TSP example to a .tsp file"""

    # Pattern to match 1 or more consecutive digits
    regex_pattern = r'(0|[1-9][0-9]*)'
    pattern = re.compile(regex_pattern)

    matches = pattern.findall(file_path)

    with open(file_path, 'w') as fp:
        if matches:
            # number of training example in dataset
            example_num = int(matches[-1]) 
            fp.write(f"NAME: example-{example_num}\n")
        else:
            fp.write("NAME: example\n")
        
        fp.write("TYPE: TSP\n")
        fp.write(f"DIMENSION: {int(len(coordinates) / 2)}\n")
        fp.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        fp.write("NODE_COORD_SECTION\n")
        for i in range(0, len(coordinates), 2):
            fp.write(f"{i // 2 + 1} {coordinates[i]} {coordinates[i + 1]}\n")
        fp.write("EOF\n")
        fp.flush()


def run_concorde(input_file, output_file):
    """
    Runs the concorde TSP solver, which provides us with an
    optimal city traversing.
    """
    print('Running concorde...')
    # command = ["concorde", "-o", output_file, "-x", input_file]
    command = ["concorde", "-v", "-o", output_file, input_file]
    subprocess.run(command)

    # result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # if result.returncode != 0:
    #     error_message = result.stderr.decode('utf-8')
    #     raise RuntimeError(f"Concorde failed with return code {result.returncode}.\nError message: {error_message}")




def main(input_folder, input_file, output_folder, output_file, output_log_path):

    with open(output_log_path, 'a') as f:
        f.write('aaa\n')

    input_file_path = os.path.join(input_folder, input_file)
    output_file_path = os.path.join(output_folder, output_file)

    unread_solution_files = 0

    with open(output_log_path, 'a') as f:
        f.write(f"Current Working Directory: {os.getcwd()}\n")
        f.write('bbb\n')
        f.write(f'{input_file_path=}\n')
        f.flush()
        
        # Read input examples of TSP's
        with open(input_file_path, 'r') as in_f:
            lines = in_f.readlines()
            f.write(f'length of lines: {len(lines)}\n')
            f.flush()

    with open(output_log_path, 'a') as f: # Open logging file

        f.write('ccc\n')
        f.flush()

        with open(output_file_path, 'a') as solutions: 

            for idx, line in enumerate(lines):
                coordinates = [int(num) for num in line.split()]
                f.write(f'example index {idx}\n')
                f.write(f'coordinates: {coordinates}\n')
                f.flush()

                tsplib_file_path = os.path.abspath(os.path.join(input_folder, f"example_{idx + 1}.tsp")) # INPUT TO CONCORDE 
                f.write(f'{tsplib_file_path=}\n')
                f.flush()
                
                write_tsplib_file(tsplib_file_path, coordinates)
                # print(f"Converted example {idx + 1} to {tsplib_file_path}")

                sol_file_path = os.path.abspath(os.path.join(output_folder, f"example_{idx + 1}.sol")) # OUTPUT FILE OF CONCORDE

                # try:
                
                run_concorde(tsplib_file_path, sol_file_path)

                # time.sleep(0.1)
                
                # except RuntimeError as e:
                #     print(f"Error: {e}")
                #     unread_solution_files += 1
                #     print(f'{unread_solution_files=}\n')

                #     # Delete created files
                #     if os.path.isfile("./" + f"example_{idx + 1}.sol"):
                #         os.remove("./" + f"example_{idx + 1}.sol")

                #     if os.path.isfile(sol_file_path):
                #         os.remove(sol_file_path) # Delete solution trajectory file
                    
                #     if os.path.isfile(tsplib_file_path):
                #         os.remove(tsplib_file_path) # Delete generated TSPLIB file

                #     continue # skip to next example
                


                # Skip examples which are not being solved and count the number of unsolved examples.
                # Skip to next example.

                f.write('ddd\n')
                f.flush()

                # Retry logic for file existence
                # max_retries = 2
                # current_retry = 0
                # while current_retry < max_retries:
                #     if os.path.isfile(sol_file_path):
                #         break  # Exit the loop if the file exists
                #     else:
                #         current_retry += 1
                #         time.sleep(0.5)


                if not os.path.isfile(sol_file_path):
                    f.write(f'File {sol_file_path} does not exist.\n')
                    f.flush()
                    unread_solution_files += 1
                    f.write(f'{unread_solution_files=}\n')
                    f.flush()
                    continue
                else:
                    f.write(f'File does exists!: {sol_file_path}\n')
                    f.flush()
                


                with open(sol_file_path, 'r') as solution_file:
                    lines = solution_file.readlines()

                    f.write(f'printing lines in solution file...\n{lines}\n')
                    f.flush()

                    if not lines: # list is empty, hence no solution found
                        f.write(f'No solution found in file {sol_file_path}\n')
                        f.flush()
                        unread_solution_files += 1
                        f.write(f'{unread_solution_files=}\n')
                        f.flush()
                        break
                    
                    # TSP solution trajectory ''should'' be stored in the last line of solution_file
                    solution_tour = lines[-1].rstrip()

                    # Convert the input city coordines into a more readible form
                    result_line, city_mapping = add_city_letters(line.rstrip())

                    # Convert the solution_tour which consists of cities denoted with number
                    # to cities denoted with letters
                    solution_tour_letters_list = [city_mapping[city] for city in solution_tour.split()]
                    solution_tour_letters = " ".join(solution_tour_letters_list)

                    # We add a trailing white space because we later use a GPT2 tokenizer
                    # The "; " separates the input city coordinates from the optimal tour
                    solutions.write(" " + result_line + "; " + solution_tour_letters + "\n")
                    solutions.flush()


                # UNCOMMENT ALL BELOW FOR FILE DELETION
                f.write('Deleting files...\n')
                f.flush()

                
                if os.path.isfile("./" + f"example_{idx + 1}.sol"):
                    try:
                        os.remove("./" + f"example_{idx + 1}.sol")
                    except FileNotFoundError:
                        pass

                # Delete files since we wrote their contents in output_file
                if os.path.isfile(sol_file_path):
                    try:
                        os.remove(sol_file_path) # Delete solution trajectory file
                    except FileNotFoundError:
                        pass
                
                if os.path.isfile(tsplib_file_path):
                    try:
                        os.remove(tsplib_file_path) # Delete generated TSPLIB file
                    except FileNotFoundError:
                        pass
                

                f.write('Deleting files using glob...\n')
                f.flush()

                # Delete files with specific extensions. These files get spawned by concorde
                extensions_to_delete = [".sav~", ".mas~", ".pul~", ".sav", ".mas", ".pul", ".sol"]
                for extension in extensions_to_delete:
                    files_to_delete = glob.glob(os.path.join("/home/os415/rds/hpc-work/concorde/TSP", f"*{extension}"))
                    for file_path in files_to_delete:
                        try:
                            os.remove(file_path)
                        except FileNotFoundError:
                            pass
                
                extensions = ['.sol', '.tsp']
                for ext in extensions:
                    files = glob.glob(os.path.join(input_folder, f'*{ext}'))  # Add * before the extension
                    for file in files:
                        try:
                            os.remove(file)
                        except FileNotFoundError:
                            pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Add arguments
    parser.add_argument('-i', '--input_file', required=True, help='Input file path. \
    The file should contain training examples as x y city coordinates for TSP.')
    
    parser.add_argument('-o', '--output_file', required=True, help='Output file path. Upon program completion, \
    will have training examples with city coordinates and optimal traversings for TSP.')

    parser.add_argument('-ifol', '--input_folder', required=True, help='Input folder that contains data')

    parser.add_argument('-olog' , '--output_log_path', help='For manually printing logs')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values
    input_file = args.input_file
    output_file = args.output_file

    input_folder = args.input_folder     #"./training-data/10M"
    output_folder = input_folder

    output_log_path = args.output_log_path # path that I log to my own print statements


    with open(output_log_path, 'a') as myfile:
        myfile.write("Hello world\n")
        myfile.write(f"Current Working Directory: {os.getcwd()}\n")
        myfile.write(f"{input_folder=}\n")
        myfile.write(f"{output_folder=}\n")
        myfile.write(f"{input_file=}\n")
        myfile.write(f"{output_log_path=}\n")


    main(input_folder, input_file, output_folder, output_file, output_log_path)
