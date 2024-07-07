import os

num_lines_per_file = 50000
input_file_path = './examples-14M.txt'

with open(input_file_path, 'r') as input_file:
    for file_number in range(1, int(1.4e7 / num_lines_per_file) + 1):

        output_folder = f'./batch_{file_number}'
        os.makedirs(output_folder, exist_ok=True)

        output_file_path = os.path.join(output_folder, f'batch_{file_number}.txt')
        
        with open(output_file_path, 'w') as f:
            lines_written = 0
            while lines_written < num_lines_per_file:
                line = input_file.readline()
                if not line:
                    break  # End of file
                f.write(line)
                lines_written += 1

        # Explicitly set permissions for the output folder and file
        os.chmod(output_folder, 0o755)  # Read and execute permissions for all
        os.chmod(output_file_path, 0o644)  # Read and write permissions for owner, read-only for others
