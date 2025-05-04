import torch
import tiktoken
from torch.utils.data import DataLoader, Dataset

# Create DataSet and DataLoader for efficient model training

class GPTDataSet(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.output_ids = []

        # Tokenize the text
        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # Sliding window approach to get the input and output pairs for training
        for i in range(0, len(tokens) - max_length, stride):
            input = tokens[i: i + max_length]
            output = tokens[i+1:i+1+max_length]
            self.input_ids.append(torch.tensor(input))
            self.output_ids.append(torch.tensor(output))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.output_ids[idx]

def create_dataloader(text, batch_size=4, max_length=256,stride=128, shuffle=False, drop_last=False, num_workers=0):

    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDataSet(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader

# Read the text and create dataloader for it
# Loads the data into batches
HarryPotter = open('HarryPotter.txt')
text = HarryPotter.read()
dataloader = create_dataloader(text, 4, 256, 128)


data_iter = iter(dataloader)

