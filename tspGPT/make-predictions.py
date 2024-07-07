import os
import pdb
import csv
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from time_in_hh_mm_ss import get_time_hh_mm_ss
from char_dataset import CharDataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import GPT2LMHeadModel


print('Parsings arguments...')
parser = argparse.ArgumentParser()

# Add mandatory arguments
parser.add_argument('-pm', '--pretrainedmodel', help='Load Pretrained model from path')
parser.add_argument('-e', '--eval_file', help='Optimal paths Eval dataset file path')
parser.add_argument('-ebc', '--eval_file_swappedBC', help='BC Swapped eval paths')
parser.add_argument('-p', '--pos', help='choose which tokens activations to store')

# Parse the command-line arguments
args = parser.parse_args()


model_path = args.pretrainedmodel # "anticlk-90%-swap-BC/checkpoint-220000"
eval_file = args.eval_file
evalBC_file = args.eval_file_swappedBC
pos = int(args.pos)

print('Parsed!')


# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'{device=}')

# Load pre-trained GPT2 model
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
trg_len = 4
n_layer = model.config.n_layer


assert pos in range(trg_len)


# Load evaluation dataset 
print(f'Reading {eval_file}...')
with open(os.path.join('./datasets/anticlk', eval_file), "r", encoding="utf-8") as f:
    eval_lines = [line.strip().split() for line in f]
print(f'Reading of {eval_file} done!')


eval_set_size = len(eval_lines)


# Read evalBC text file and split it into lines
print(f'Reading {evalBC_file}...')
with open(os.path.join('./datasets/anticlk', evalBC_file), "r", encoding="utf-8") as f:
    evalBC_lines = [line.strip().split() for line in f]
print(f'Reading of {evalBC_file} done!')

# Length of example sequence
max_length = len(eval_lines[0])

# Length of input sequence
input_len = max_length - trg_len


assert len(eval_lines[0]) == len(evalBC_lines[0])
assert len(eval_lines) == len(evalBC_lines)

# Create a dictionary with a key "text" and values as the lines from the file
tokenized_eval_lines = CharDataset(eval_lines)
dataset_eval_dict = {"input_ids": tokenized_eval_lines, "attention_mask": np.ones_like(tokenized_eval_lines).tolist()}


tokenized_evalBC_lines = CharDataset(evalBC_lines)
labelsBC = np.zeros((len(evalBC_lines), trg_len), dtype=int)

# Loop over eval examples and store last trg_len letters
for i in range( len(evalBC_lines) ):
    labelsBC[i] = tokenized_evalBC_lines[i][-trg_len:]

labelsBC = torch.tensor( labelsBC )

del tokenized_evalBC_lines
del evalBC_lines


# Convert the dictionary to a Hugging Face Dataset
hf_eval_dataset = Dataset.from_dict(dataset_eval_dict)
print('Converted to HuggingFace Dataset dictionary!')

dataset_eval_dict = hf_eval_dataset

print('Adding target labels...')
# The model should only predict last "trg_len" tokens. Set other tokens to -100.


def add_labels(examples):
    
    labels = examples["input_ids"].copy()
    # array_of_labels = np.array(lables)

    # Below line does this: labels[:, :-trg_len] = -100
    target_labels = [[-100] * (len(row) - trg_len) + row[-trg_len:] for row in labels]
    examples['labels'] = target_labels

    return examples

lm_eval_dataset = dataset_eval_dict.map(
    add_labels,
    batched=True,
    num_proc=32,
)




# Hooks
acts_dict = {'attn': {}, 'mlp': {}, 'block': {}} # stores attention, mlp and overall block layers output
def getActivation(name):
    """
    Function to define hooks for storing activations at different layers in the model.
    """
    def hook(model, input, output): # input is input of a layer (e.g. of attn layer), output is output of a layer
        if 'attn' in name:
            acts_dict['attn'][name] = output[0][:, -1, :].detach().clone() # detach the gradient and clone
        elif 'mlp' in name:
            acts_dict['mlp'][name] = output[0][:, -1, :].detach().clone()
        elif 'block' in name:
            acts_dict['block'][name] = output[0][:, -1, :].detach().clone()
        else:
            raise ValueError('Unknown layer type')
    return hook


# Now we need to assign a hook to different layers of the model
def gethooks(model, to_probe):
    """
    Get hooks for the model
    """
    hook_handles = []

    # Attention
    if to_probe == 'attn':
        # Create a list of hook handles
        for i in range(len(model.transformer.h)):
            hook_handles.append(model.transformer.h[i].attn.register_forward_hook(getActivation('attn' + str(i))))

    # MLPs
    elif to_probe == 'mlp':
        for i in range(len(model.transformer.h)):
            hook_handles.append(model.transformer.h[i].mlp.register_forward_hook(getActivation('mlp' + str(i))))

    # Overall blocks
    elif to_probe == 'block':
        for i in range(len(model.transformer.h)):
            hook_handles.append(model.transformer.h[i].register_forward_hook(getActivation('block' + str(i))))

    return hook_handles


# Function to generate predictions for each example in the evaluation dataset using batching
def generate_predictions(tokenized_lines, model, bs=2048):
    """
    Generate predictions for each example in the evaluation dataset using batching.

    Args:
        tokenized_lines (list): The tokenized input data.
        model (object): The model used for generating predictions.
        bs (int, optional): Batch size for processing. Default is 2048.

    Returns:
        dict: 
            acts_opt: Stores tensors of activations for optimally (shortest TSP path) solved examples.
            acts_BC: Stores tensors of activations for BC swapped examples.
        list:
            letters_opt: Stores the first predicted letter for an optimally solved example.
            letters_BC: Stores the first predicted letter for a BC swapped solved example.
    """
    
    # Initialisation
    acts_opt = {}
    acts_BC = {}
    letters_opt = [] 
    letters_BC = []


    to_probe = "block"  # You can choose which part of the model you want to probe

    # Add keys and empty lists to the dictionary
    for i in range(n_layer):
        key = f"{to_probe}{i}"
        acts_opt[key] = []
        acts_BC[key] = []
    
    
    # Get input_ids and target labels
    input_ids = tokenized_lines['input_ids']
    labels = np.array(tokenized_lines['labels'])
    labels_trglen = torch.tensor(labels[:, -trg_len:])
    labelsBC_trglen = labelsBC[:, -trg_len:]

    # Arrays to store predictions as:
    predictions = [] # letters & numbers 
    pred_ids = [] # token ids
    exact_match = 0.
    BC_exact_match = 0.

    # tensor to store the scores(logits) of shape (27000,trglen,vocab_size)
    logits_trglen = torch.zeros(len(input_ids), trg_len, model.config.vocab_size)

    total_iterations = len(input_ids)
    num_batches = (total_iterations + bs - 1) // bs  # Calculate number of batches

    # Initialize tqdm manually
    progress_bar = tqdm(total=total_iterations, desc="Generating Predictions", unit="lines")


    # Assign the hooks
    hook_handles = gethooks(model, to_probe)


    # Method to store a hooked activation of example idx, but only if less that MAX_SIZE are stored
    def store_activations(acts_dict, acts_opt, idx, letters, predicted_letter, MAX_SIZE=100):
        """
        Args:
            acts_dict (dict): Dictionary storing hooked activations.
            acts_opt (dict): Output dictionary of up to MAX_SIZE stored hooked activations.
            letters (list): stores predicted first letters for previous examples
            predicted_letter (str): the first TSP letter the model predicted for example 'idx'
            idx: index of the example whose activations we are storing
            MAX_SIZE (int): Number of activation tensors to store per block.
        """

        # Check if all lists of acts_opt have the same length
        assert len(set(len(lst) for lst in acts_opt.values())) == 1, "All lists must have the same length."

        # Find the maximum length of the lists in acts_opt
        max_len = max(len(lst) for lst in acts_opt.values())

        # Store up to MAX_SIZE entries
        if max_len >= MAX_SIZE:
            return


        # Loop over hooked activations
        for k, v in acts_dict[to_probe].items():

            # Extract hooked activations tensor for example "idx"
            acts_opt[k].append(v[idx].cpu().numpy())
        
        # Store predicted letter to list
        letters.append(predicted_letter)

    
    # Eval loop
    for batch_start in range(0, total_iterations, bs):
        batch_end = min(batch_start + bs, total_iterations)
        batch_input_ids = torch.tensor(input_ids[batch_start:batch_end]).to(device)
        
        with torch.no_grad():
            # Reinitialise to make acts_dict[to_probe] empty

            # Model makes (token) predictions 
            if pos == 0:

                acts_dict[to_probe] = {}

                output = model.generate(batch_input_ids[:, :-trg_len], max_length=max_length, num_return_sequences=1, pad_token_id=-100, output_scores=True, return_dict_in_generate=True)

            else: 

                first_output = model.generate(batch_input_ids[:, :-trg_len], max_length=input_len+pos, num_return_sequences=1, pad_token_id=-100, output_scores=True, return_dict_in_generate=True)
                # concatenate ouput and

                # pdb.set_trace()

                acts_dict[to_probe] = {} # has to reset. move between first_output and output

                
                # Predict the rest of the sequence
                output = model.generate(first_output.sequences, max_length=max_length, num_return_sequences=1, pad_token_id=-100, output_scores=True, return_dict_in_generate=True)

                # pdb.set_trace()
                
        if pos == 0:

            # Store scores (logits) of size (bs,trglen,vocab_size)
            logits_bs = torch.stack(output.scores, dim=1)
        
        else:

            scores_list = [first_output.scores[i] for i in range(len(first_output.scores))] + \
                          [output.scores[i] for i in range(len(output.scores))]
            
            

            # scores_list = [first_output.scores[0],
            #                 first_output.scores[1],
            #                 first_output.scores[2],
            #                 output.scores[0]]
            
            # Extract the entries from output.scores and convert the tuple to a list
            # scores_list = list(output.scores)

            # Prepend the first entry from first_output.scores[0]
            # scores_list.insert(0, first_output.scores[0])
            # scores_list.insert(1, first_output.scores[1])
            # scores_list.insert(2, first_output.scores[2])


            # Store scores (logits) of size (bs,trglen,vocab_size)
            logits_bs = torch.stack(scores_list, dim=1)


        logits_trglen[batch_start:batch_end] = logits_bs


        # predicted tokenized sequence
        preds = output.sequences.cpu().numpy()

        # Convert preds to binary (0 or 1). 1 if prediction is correct
        token_predictions = np.equal(preds[:, -trg_len:], labels[batch_start:batch_end, -trg_len:])
        BC_token_predictions = np.equal(preds[:, -trg_len:], labelsBC[batch_start:batch_end].numpy())        
        token_predictions = np.all(token_predictions, axis=1).astype(float)
        BC_token_predictions = np.all(BC_token_predictions, axis=1).astype(float)
        exact_match += token_predictions.sum()
        BC_exact_match += BC_token_predictions.sum()

        for idx in range(batch_end - batch_start):

            o_ids = preds[idx]
            pred_ids.append(o_ids)

            # Use stoi to decode sequence of ids
            generated_text = tokenized_eval_lines.decode_sequence(o_ids)

            # First and Second Predicted Letters
            letter_to_store = generated_text.split()[-trg_len + pos] 
            
            if token_predictions[idx]: # Model made correct prediction

                # Write to output
                predictions.append(generated_text.rstrip() + ' CORRECT\n')                

                store_activations(acts_dict, acts_opt, idx, letters_opt, letter_to_store)
            
            else: # Model made incorrect prediction

                # Write to output
                predictions.append(generated_text.rstrip() + ' WRONG\n')

                store_activations(acts_dict, acts_BC, idx, letters_BC, letter_to_store)

                
        # Update progress bar
        progress_bar.update(batch_end - batch_start)
    

    # After generating predictions, remove the hooks
    for handle in hook_handles:
        handle.remove()

    logits_trglen = logits_trglen.reshape(-1, logits_trglen.shape[-1])

    # inherently converts logits to softmax probabilities
    nll = F.cross_entropy(
        logits_trglen,
        labels_trglen.reshape(-1),
        ignore_index=-100, 
        reduction='mean',
        )
    
    nll_BC = F.cross_entropy(
        logits_trglen,
        labelsBC.reshape(-1),
        ignore_index=-100, 
        reduction='mean',
    )


    # Compute nll on each token separately
    logits_trglen = logits_trglen.reshape(eval_set_size, trg_len, -1)
    
    # Initialise empty arrays
    per_token_nll = torch.zeros(trg_len)

    if evalBC_file is not None:
        per_token_nll_BC = torch.zeros(trg_len)

    for i in range(trg_len):
    
        per_token_nll[i] = F.cross_entropy(
            logits_trglen[:,i,:],
            labels_trglen[:, i],
            ignore_index=-100, 
            reduction='mean',
            ).item()
        
        if evalBC_file is not None:
            per_token_nll_BC[i] = F.cross_entropy(
                logits_trglen[:,i,:],
                labelsBC_trglen[:, i],
                ignore_index=-100, 
                reduction='mean',
                ).item()

    # Close progress bar manually
    progress_bar.close()

    # Create a dictionary to store the variables
    data_to_save = {
        'nll_opt': nll,
        'nll_BC': nll_BC,
        'token0_nll_opt': per_token_nll[0],
        'token0_nll_BC': per_token_nll_BC[0],
        'token1_nll_opt': per_token_nll[1],
        'token1_nll_BC': per_token_nll_BC[1],
        'token2_nll_opt': per_token_nll[2],
        'token2_nll_BC': per_token_nll_BC[2],
        'token3_nll_opt': per_token_nll[3],
        'token3_nll_BC': per_token_nll_BC[3]
    }


    # Save numbers to pickle

    # # Specify the filename
    # fn = model_path.split('/')[0] + '-' + model_path.split('/')[1].split('-')[1]
    # filename = f'train-{fn}.pkl'
    # folder_path = '/home/os415/rds/hpc-work/tspGPT/five-nll-plot-train-set'

    # # Open the file in binary write mode and use pickle to save the data
    # with open(os.path.join(folder_path,filename), 'wb') as file:
    #     pickle.dump(data_to_save, file)

    # print(f"Variables saved to {filename}")

    

    exact_match /= total_iterations
    BC_exact_match /= total_iterations
    print(f'{exact_match=:.7f}')
    print(f'{BC_exact_match=:.7f}')
    print(f'{nll=:.7f}')
    print(f'{nll_BC=:.7f}')

    return predictions, pred_ids, acts_opt, acts_BC, letters_opt, letters_BC 

# Generate predictions
eval_predictions, eval_pred_ids, acts_opt, acts_BC, letters_opt, letters_BC = generate_predictions(lm_eval_dataset, model)


# Define the file and folder to save to

# Extract the model name
model_name = model_path.split('/')[0]

# Define the folder name
folder_name = os.path.join('./datasets/anticlk/inference', model_name)

# Create the directory if it doesn't exist
os.makedirs(folder_name, exist_ok=True)

file_name = f"V3.1_letter{pos}_" + model_name + ".txt"
pkl_acts_opt = f"V3.1_letter{pos}_acts_opt_" + model_path.split('/')[0] + ".pkl"
pkl_acts_BC = f"V3.1_letter{pos}_acts_BC_" + model_path.split('/')[0] + ".pkl"
pkl_letters_opt = f"V3.1_letter{pos}_letters_opt_" + model_path.split('/')[0] + ".pkl"
pkl_letters_BC = f"V3.1_letter{pos}_letters_BC_" + model_path.split('/')[0] + ".pkl"

# # Open the file in write mode
# with open(os.path.join(folder_name, file_name), 'w') as f:
#     # Write each element of the list as a separate line in the text file
#     for line in eval_predictions:
#         f.write(line)


# pdb.set_trace()

# # Save acts_opt to a file
# with open(os.path.join(folder_name, pkl_acts_opt), 'wb') as f:
#     pickle.dump(acts_opt, f)

# # Save acts_BC to a file
# with open(os.path.join(folder_name, pkl_acts_BC), 'wb') as f:
#     pickle.dump(acts_BC, f)

# # Save letters_opt to a file
# with open(os.path.join(folder_name, pkl_letters_opt), 'wb') as f:
#     pickle.dump(letters_opt, f)

# # Save letters_BC to a file
# with open(os.path.join(folder_name, pkl_letters_BC), 'wb') as f:
#     pickle.dump(letters_BC, f)

# # Confirmation message
# print(f"List saved to {file_name}.\nPickled files saved to {pkl_acts_opt}\nand {pkl_acts_BC}")
# print(f'Saved letters to {pkl_letters_opt} and {pkl_letters_BC}')
