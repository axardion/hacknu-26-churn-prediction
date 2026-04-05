import csv
import os

input_file = "data/preprocessed/train/train_users_generations.csv"
temp_file = "data/preprocessed/train/train_users_generations_temp.csv"

# Open the input file for reading AND the temp file for writing at the same time
with open(input_file, "r", encoding="utf-8") as infile, \
     open(temp_file, "w", encoding="utf-8", newline="") as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Process the header
    header = next(reader)
    writer.writerow(header)
    
    cgi = header.index("generation_id")
    
    # Process line-by-line (streams the data, uses almost zero memory)
    for row in reader:
        # Check if the row has actual data (ignores blank lines/newlines)
        if any(cell.strip() for cell in row):
            row[cgi] = ""
            writer.writerow(row)

# Safely replace the original file with the newly cleaned file
os.replace(temp_file, input_file)

print("Processing complete!")