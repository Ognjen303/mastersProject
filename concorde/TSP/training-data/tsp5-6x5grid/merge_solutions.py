import os

def concatenate_solutions(folder_path, num_batches, output_file):
    with open(output_file, 'w') as output:
        for batch_num in range(1, num_batches + 1):
            batch_folder = os.path.join(folder_path, f'batch_{batch_num}')
            solutions_file = os.path.join(batch_folder, f'solutions_{batch_num}.txt')

            try:
                with open(solutions_file, 'r') as solutions:
                    output.writelines(solutions.readlines())
            except FileNotFoundError:
                print(f"Warning: {solutions_file} not found.")

if __name__ == "__main__":
    folder_path = "/home/os415/rds/hpc-work/concorde/TSP/training-data/tsp5-6x5grid"
    num_batches = 267
    output_file = os.path.join(folder_path, 'tsp5-6x5grid-14M.txt')

    concatenate_solutions(folder_path, num_batches, output_file)
    print(f"All solutions concatenated and saved to {output_file}.")
