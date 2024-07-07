import os
import pdb
import time
import argparse
import numpy as np
import wandb
import torch
import torch.nn.functional as F
from tqdm import tqdm
from time_in_hh_mm_ss import get_time_hh_mm_ss
from char_dataset import CharDataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


parser = argparse.ArgumentParser()

# Add mandatory arguments
parser.add_argument('-i', '--input_file', help='Input dataset file path', default='solutions_1k.txt')
parser.add_argument('-e', '--eval_file', help='Eval dataset file path', default='C00.txt')
parser.add_argument('-o', '--output_dir', help='Name of output directory to save model weights', default='TSP-10cities-1k-examples')
parser.add_argument('-ch', '--checkpoint', help='Resume training from checkpoint', default=None)
parser.add_argument('-ebc', '--eval_file_swappedBC', help='Eval dataset with swapped B and C', default=None)
parser.add_argument('-eran', '--eval_file_random', help='Eval dataset random', default=None)
parser.add_argument('-lr', '--learning_rate', help='Learning rate of the model', default='1e-4')
parser.add_argument('-wd', '--weightdecay', help='Weight decay of the model', default='1e-4')
parser.add_argument('-epochs', '--epochs', help='Number of epochs to train model for', default='10')
parser.add_argument('-trglen', '--target_length', help='Target Length of the sequence the model needs to predict. \
                    cities remaining to traverse (assuming we start traversing from city A)', default='9')
parser.add_argument('-msize', '--model_size', help='Size of GPT2 that we will use', default='1542')


# Parse the command-line arguments
args = parser.parse_args()

input_file = args.input_file
eval_file = args.eval_file
output_dir = args.output_dir
checkpoint = args.checkpoint
evalBC_file = args.eval_file_swappedBC
eval_random_file = args.eval_file_random
learning_rate = float(args.learning_rate)
weight_decay = float(args.weightdecay)
num_training_epochs = float(args.epochs)
trg_len = int(args.target_length)
model_size = int(args.model_size)

print(f'{learning_rate=}')
print(f'{weight_decay=}')
print(f'{num_training_epochs=}')
print(f'{trg_len=}')
print(f'{eval_file=}')
print(f'{evalBC_file=}')
print(f'{eval_random_file=}')

print(f'Number of parameters in GPT2 model: {model_size}M')

# GPT2 Model size(millions of parameters): (n_embd, n_layer, n_head) 
model_parameters = {
    3: (48, 3, 3),    # For nanoGPT
    7: (128, 4, 4),   # For microGPT
    12: (192, 6, 6),  # For miniGPT
    117: (780, 12, 12)  # GPT2
}

# TODO: Change 780 to 768 in GPT2 model_parameters



model_training_params = {
    # Batch sizes for train and eval sets, eval every "x" steps eval_steps=800 
    3: (65536, 65536, 25),    # For nanoGPT
    7: (65536, 65536, 25),   # For microGPT
    12: (16384, 16384, 200),  # For miniGPT
    117: (2048, 2048, 800)    # GPT2
}

if model_size in model_parameters and model_size in model_training_params:
    n_embd, n_layer, n_head = model_parameters[model_size]
    bs_train, bs_eval, eval_steps = model_training_params[model_size]
else:
    raise ValueError("Invalid model size specified.")


acc_steps_train=1       # Gradient accumulation steps
acc_steps_eval=1  



# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'{device=}')


# Train set text file and split into lines
print(f'Reading {input_file}...')
with open(os.path.join('./datasets/anticlk', input_file), "r", encoding="utf-8") as f:
    lines = [line.strip().split() for line in f]
print(f'Reading of {input_file} done!')


# Number of characters in one problem definition
max_length = len(lines[0]) # model.config.n_positions # 1024


# Eval set text file is read and split it into lines
print(f'Reading {eval_file}...')
with open(os.path.join('./datasets/anticlk', eval_file), "r", encoding="utf-8") as f:
    eval_lines = [line.strip().split() for line in f]
print(f'Reading of {eval_file} done!')


eval_set_size = len(eval_lines)

assert max_length == len(eval_lines[0])


if evalBC_file is not None:

    # Eval set with cities B and C swapped in optimal trajectory
    print(f'Reading {evalBC_file}...')
    with open(os.path.join('./datasets/anticlk', evalBC_file), "r", encoding="utf-8") as f:
        evalBC_lines = [line.strip().split() for line in f]
    print(f'Reading of {evalBC_file} done!')

    assert max_length == len(evalBC_lines[0])
    assert len(evalBC_lines) == eval_set_size


if eval_random_file is not None:

    # Eval set with cities two cities randomly swapped in optimal trajectory
    # (used for sanity check only), any predictions on this will be bad
    print(f'Reading {eval_random_file}...')
    with open(os.path.join('./datasets/anticlk', eval_random_file), "r", encoding="utf-8") as f:
        eval_random_lines = [line.strip().split() for line in f]
    print(f'Reading of {eval_random_file} done!')

    assert max_length == len(eval_random_lines[0])
    assert len(eval_random_lines) == eval_set_size






if trg_len == 9:
    assert max_length == 41
elif trg_len == 8:
    assert max_length == 37 
elif trg_len == 6:
    assert max_length == 29
elif trg_len == 5:
    assert max_length == 25
elif trg_len == 4:
    assert max_length == 21


print('Converting to HuggingFace Dataset dictionary...')

# Create a dictionary with a key "text" and values as the lines from the file
tokenized_lines = CharDataset(lines) # shuffled_lines
tokenized_eval_lines = CharDataset(eval_lines)

if evalBC_file is not None:
    
    tokenized_evalBC_lines = CharDataset(evalBC_lines)
    
    # Initialise arrays to store target labels
    labelsBC_trglen = np.zeros((len(evalBC_lines), trg_len), dtype=int)

    # Loop over eval examples and store last trg_len tokenised letters
    for i in range( len(evalBC_lines) ):
        labelsBC_trglen[i] = tokenized_evalBC_lines[i][-trg_len:]
    
    # Convert to tensors
    labelsBC_trglen = torch.tensor( labelsBC_trglen )

    # Delete stuff we don't need anymore
    del tokenized_evalBC_lines, evalBC_lines


if eval_random_file is not None:

    tokenized_eval_random_lines = CharDataset(eval_random_lines)

    labelsRandom_trglen = np.zeros((len(eval_random_lines), trg_len), dtype=int)

    # Loop over eval examples and store last trg_len tokenised letters
    for i in range( len(eval_random_lines) ):
        labelsRandom_trglen[i] = tokenized_eval_random_lines[i][-trg_len:]

    # Convert to tensors
    labelsRandom_trglen = torch.tensor( labelsRandom_trglen )

    # Delete stuff we don't need anymore
    del tokenized_eval_random_lines, eval_random_lines



# Create HuggingFace train and eval sets
dataset_dict = Dataset.from_dict({"input_ids": tokenized_lines, "attention_mask": np.ones_like(tokenized_lines).tolist()})
dataset_eval_dict = Dataset.from_dict({"input_ids": tokenized_eval_lines, "attention_mask": np.ones_like(tokenized_eval_lines).tolist()})
print('Converted to HuggingFace Dataset dictionary!')


print('Adding target labels...')
# The model should only predict last "trg_len" tokens. Set other tokens to -100.
def add_labels(examples):
    
    labels = examples["input_ids"].copy()

    # Below line does this: labels[:, :-trg_len] = -100
    target_labels = [[-100] * (len(row) - trg_len) + row[-trg_len:] for row in labels]
    examples['labels'] = target_labels

    return examples

# Add target labels via map()
lm_tsp_dataset = dataset_dict.map(
    add_labels,
    batched=True,
    num_proc=32,
)

lm_eval_dataset = dataset_eval_dict.map(
    add_labels,
    batched=True,
    num_proc=32,
)

lm_tsp_dataset.set_format("torch")
lm_eval_dataset.set_format("torch")
print('Added target labels!')



# Define the model and training
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer

# CustomTrainer is based on https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/trainer#transformers.Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        # forward pass
        outputs = model(**inputs)

        # Assume loss is calculated internaly during training
        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, 
        # because it internally shifts the labels to the left by 1.
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
    
    
    # Run evaluation and return metrics
    def evaluate(self, eval_dataset = None, ignore_keys=None, metric_key_prefix: str = "eval"):

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        input_ids = self.eval_dataset['input_ids']
        labels = self.eval_dataset['labels']

        # Only use last trg_len entries as labels. Discards '-100' elements from labels.
        labels_trglen = labels[:, -trg_len:]

        # Initialise array to store predictions
        preds_trglen = np.zeros((eval_set_size, trg_len)) # letters & numbers

        # Tensor to store the scores(logits) of shape (eval_set_size,trglen,vocab_size)
        logits_trglen = torch.zeros(eval_set_size, trg_len, model.config.vocab_size)

        # Determine eval loop batch parameters
        total_iterations = eval_set_size
        bs = self.args.per_device_eval_batch_size
        num_batches = (total_iterations + bs - 1) // bs  # Calculate number of batches

        
        # Initialize tqdm to have a progress bar
        progress_bar = tqdm(total=total_iterations, desc="Generating Predictions", unit="lines")

        # Evaluation loop
        for batch_start in range(0, total_iterations, bs):
            batch_end = min(batch_start + bs, total_iterations)
            batch_input_ids = input_ids[batch_start:batch_end].to(device)

            # Model makes (token) predictions
            with torch.no_grad():
                output = model.generate(input_ids=batch_input_ids[:, :-trg_len], max_length=max_length, num_return_sequences=1, pad_token_id=-100, output_scores=True, return_dict_in_generate=True)

            # Store predicted tokenized sequence
            preds = output.sequences.cpu().numpy()

            # Only care about the trg_len
            preds_trglen[batch_start:batch_end] = preds[:, -trg_len:]
            
            # Store scores (logits) of size (bs,trglen,vocab_size)
            logits_bs = torch.stack(output.scores, dim=1)
            logits_trglen[batch_start:batch_end] = logits_bs

            # Update progress bar
            progress_bar.update(batch_end - batch_start)
        

        # Compute metrics
        metrics = self.compute_metrics(preds_trglen, logits_trglen, labels_trglen)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics


# Computes metrics on evaluation dataset
def compute_metrics(preds_trglen, logits_trglen, labels_trglen):
    """
    Arguments:

    preds_trglen  (numpy.array)  -> predictions made by model (a tokenized sequence) of shape (27000,trglen)
    logits_trglen (torch.Tensor) -> scores(logits) of shape (27000,trglen,vocab_size)
    labels_trglen (torch.Tensor) -> labels of shape (27000,trglen)
    """

    curr_time = time.time()
    get_time_hh_mm_ss(curr_time - start_time)

    # Get the logits and then compute CE of logits w.r.t. GT labels
    # NOTE: I removed the label shifting since I only look at last trglen 
    # indices anyway.

    logits_trglen = logits_trglen.reshape(-1, logits_trglen.shape[-1])

    # inherently converts logits to softmax probabilities
    nll = F.cross_entropy(
            logits_trglen,
            labels_trglen.reshape(-1),
            ignore_index=-100, 
            reduction='mean',
            )

    if evalBC_file is not None:
    
        nll_BC = F.cross_entropy(
                logits_trglen,
                labelsBC_trglen.reshape(-1),
                ignore_index=-100, 
                reduction='mean',
                )
    
    if eval_random_file is not None:

        nll_ran = F.cross_entropy(
                logits_trglen,
                labelsRandom_trglen.reshape(-1),
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



    # Calculate exact_match and per_token_accuracy

    # Convert seqs to binary (0 or 1). 1 if prediction is correct
    correct_preds = np.equal(preds_trglen, labels_trglen).numpy()

    # Compute per token accuracies
    per_token_accuracy = 1.0 * correct_preds.sum() / correct_preds.size

    correct_preds = np.all(correct_preds, axis=1).astype(float)


    # Average exact match over eval dataset
    exact_match = np.mean(correct_preds).item()


    if evalBC_file is not None:
        BC_correct_preds = np.equal(preds_trglen, labelsBC_trglen).numpy()
        BC_per_token_accuracy = 1.0 * BC_correct_preds.sum() / BC_correct_preds.size
        BC_correct_preds = np.all(BC_correct_preds, axis=1).astype(float)
        BC_exact_match = np.mean(BC_correct_preds).item()



    log_dict = {
        "exact_match": exact_match,
        "per_token_accuracy": per_token_accuracy,
        "nll_wrt_opt_path": nll.item(),
        "token0_nll_wrt_opt_path": per_token_nll[0].item(),
        "token1_nll_wrt_opt_path": per_token_nll[1].item(),
        "token2_nll_wrt_opt_path": per_token_nll[2].item(),
        "token3_nll_wrt_opt_path": per_token_nll[3].item()
    }

    if evalBC_file is not None:
        log_dict.update({
            "exact_match_wrt_BC_swapped": BC_exact_match,
            "per_token_accuracy_wrt_BC_swapped": BC_per_token_accuracy,
            "nll_wrt_B&C_swapped": nll_BC.item(),
            "token0_nll_wrt_B&C_swapped": per_token_nll_BC[0].item(),
            "token1_nll_wrt_B&C_swapped": per_token_nll_BC[1].item(),
            "token2_nll_wrt_B&C_swapped": per_token_nll_BC[2].item(),
            "token3_nll_wrt_B&C_swapped": per_token_nll_BC[3].item()
        })

    if eval_random_file is not None:
        log_dict.update({"nll_wrt_2_cities_randomly_swapped": nll_ran.item()})

    
    wandb.log(log_dict)


    return {"eval_loss": nll.item()}


# Initialising a GPT2 configuration
configuration = GPT2Config(
    vocab_size=13,  # Default vocabulary size for GPT-2 large
    n_positions=1024,  # Default maximum position embeddings
    n_embd=n_embd,       # Dimension of the model (equivalent to d_model)
    n_layer=n_layer,        # Number of transformer layers
    n_head=n_head,         # Number of attention heads
)

# Initializing a model (with random weights) from the configuration
model = GPT2LMHeadModel(configuration)
model.to(device)


training_args = TrainingArguments(
    output_dir = output_dir,
    report_to='wandb',
    # save_strategy='no',
    logging_steps=eval_steps,
    save_steps=eval_steps,  # Set to a large number or comment out to avoid saving checkpoints during training
    save_total_limit = 2, # Set to 0 to disable checkpoint saving. Only last 5 models are saved. Older ones are deleted. 
    load_best_model_at_end = True,
    log_level='info',
    evaluation_strategy = 'steps',
    eval_steps=eval_steps,
    num_train_epochs = num_training_epochs,
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    #auto_find_batch_size=True,
    per_device_train_batch_size=bs_train,   # Adjust based on GPU memory. Best for GPT2 was 512
    per_device_eval_batch_size=bs_eval,     # Adjust based on GPU memory. Best for GPT2 was 512
    gradient_accumulation_steps=acc_steps_train,   # Accumulates gradients in smaller steps, then run model's optimisation step. Best left commented out for GPT2.
    eval_accumulation_steps=acc_steps_eval,        # Set a number of steps after which your predictions are sent back from GPU to the CPU (slower but uses less device memory). This should avoid your OOM. Best for GPT2 was 7.
    gradient_checkpointing=False,     # Saves strategically selected activations throughout the computational graph so only a fraction of the activations need to be re-computed for the gradients
    fp16=True,                       # Mixed precision training, saves activations in half (16-bit) precision
)


# Initialize W&B
os.environ["WANDB_API_KEY"] = "9b75cbeeec2d1e3047f812a56cb3368ea3ca6e01"
wandb.init(project="Final-countdown", name=output_dir) # Loss-futher-down

# Log initial set of parameters to W&B
wandb.config.update(training_args)

# Print parameters to console
for arg, value in training_args.__dict__.items():
    print(f'{arg}: {value}')




# Start time measurement for training
start_time = time.time()

trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset=lm_tsp_dataset,
    eval_dataset=lm_eval_dataset,
    compute_metrics=compute_metrics,
)


# Training
if checkpoint is not None:
    # If checkpoint is specified, load the model weights from that checkpoint and continue training
    trainer.train("./" + output_dir + "/checkpoint-" + checkpoint)
else:
    # If checkpoint is not specified, start training from scratch
    trainer.train()

# Training ended
end_time = time.time()
elapsed_time = end_time - start_time
print('Training finished.')
get_time_hh_mm_ss(elapsed_time)


# Save the final trained model
model.save_pretrained("./" + output_dir)

# Close W&B at the end
wandb.finish()
