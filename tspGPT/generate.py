import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from char_dataset import CharDataset
from transformers import GPT2LMHeadModel, GPT2Config, GenerationConfig

parser = argparse.ArgumentParser()

# Add mandatory arguments
parser.add_argument('-pm', '--pretrainedmodel', help='Load Pretrained model from path')

# Parse the command-line arguments
args = parser.parse_args()

model_path = args.pretrainedmodel

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'{device=}')

# Define a modified GPT2 model that includes prediction scores
class GPT2WithScores(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.output_scores = True  # Flag to indicate whether to output scores
    
    def forward(self, input_ids=None, **kwargs):
        outputs = super().forward(input_ids, **kwargs)
        if self.output_scores:
            return outputs, outputs[0]  # Return both the regular outputs and prediction scores
        else:
            return outputs


# Load pre-trained GPT2 model
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)


generation_config = GenerationConfig(
    output_scores=True
)


# Create a valid tokenizer
# Load evaluation dataset
eval_file = "eval-swapped-BC.txt"
print(f'Reading {eval_file}...')
with open(os.path.join('./datasets/anticlk', eval_file), "r", encoding="utf-8") as f:
    eval_lines = [line.strip().split() for line in f]
print(f'Reading of {eval_file} done!')

tokenizer = CharDataset(eval_lines)


trg_len = 4

# Function to generate text based on user input
def generate_output(prompt_text, model, device):

    pt = prompt_text.strip().split()
    text_len = len(pt)

    #charData = CharDataset(pt)
    #tok_ip = [charData[i][0] for i in range(text_len)]
    tok_ip = [tokenizer.stoi[s] for s in pt]

    input_ids = torch.tensor( tok_ip).to(device)
    input_ids = input_ids.unsqueeze(0) # add a batch dimension of 1
    inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_length=text_len+trg_len, num_return_sequences=1, pad_token_id=-100, output_scores=True, return_dict_in_generate=True)
    tok_op = output.sequences[0].cpu().numpy()

    logits_trglen = torch.cat(output.scores, dim=0) # tensor.Size([4,50257])
    preds_trglen = output.sequences[:,-trg_len:].reshape(-1)

    # erm... this is incorrect 
    nll = F.cross_entropy(
            logits_trglen,
            preds_trglen, # this should say labels_trglen. No point comparing your predicitons with our logits in CE....
            ignore_index=-100, 
            reduction='mean',
            )

    print(f'nll: {nll:.4f}')

    return tokenizer.decode_sequence(tok_op)

# Loop to continuously generate text based on user input
while True:
    # Get user input
    user_input = input("You: ")

    # Generate output
    output = generate_output(user_input, model, device)

    # Print generated output
    print("Model:", output)

    # Ask if the user wants to continue or exit
    cont = input("Continue? (y/n): ")
    if cont.lower() != "y":
        break
