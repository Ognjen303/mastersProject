def check_overlap(eval_file, train_file):
    eval_set = set()
    overlap_found = False
    
    with open(eval_file, 'r', encoding='utf-8') as eval_f:
        for line in eval_f:
            eval_set.add(line.strip())

    with open(train_file, 'r', encoding='utf-8') as train_f:
        for line in train_f:
            if line.strip() in eval_set:
                print("Found overlapping line in train file:", line.strip())
                overlap_found = True

    if not overlap_found:
        print(f"No overlapping lines found between {eval_file} and {train_file}.")
    else:
        print(f"Overlap found between {eval_file} and {train_file}.")

eval_file = "./new-anticlk-all-45M/shuffled.txt"
train_file = "./new-anticlk-all-45M/anticlk-all-45M.txt"
check_overlap(eval_file, train_file)
