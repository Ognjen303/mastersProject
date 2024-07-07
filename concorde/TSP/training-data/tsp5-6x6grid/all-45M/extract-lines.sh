# Define file paths
input_file="tsp5-6x6grid-all-45M.txt"
output_file="eval-27k-from-all-45M.txt"
remaining_file="remaining-45M.txt"

# Count the number of lines in the input file
num_lines=$(wc -l < "$input_file")

# Define the number of lines to extract
num_lines_to_extract=27000

# Generate a random sample of line numbers
shuf -i 1-"$num_lines" -n "$num_lines_to_extract" | sort -n > tmp_lines.txt

# Extract lines based on the random sample
awk 'NR==FNR{a[$1];next} FNR in a' tmp_lines.txt "$input_file" > "$output_file"

# Extract remaining lines
grep -vxFf "$output_file" "$input_file" > "$remaining_file"

# Clean up temporary file
# rm tmp_lines.txt

echo "Extraction complete."
