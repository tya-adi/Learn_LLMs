import torch

# Tensor of tokens

input_ids = torch.tensor([1, 2, 3, 4])

# Here there are five tokens and each token is
# represented by a four dimensional vector

vocab_size = 5
output_dim = 4

# Initializes an embedding layer with random values
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)
print(embedding_layer.weight[0])

print(embedding_layer(torch.tensor([1])))

# Final embedding layer is prepared during training phase of the model


# Similarly we create position embedding with context_length X output dimensions

context_length = 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

# Above sample code initialises a random pos embedding which we can change during training

# Input embedding = Position embedding + Vector embedding
