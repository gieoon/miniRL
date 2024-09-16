import pandas as pd
import torch
import torch.nn as nn

df = pd.DataFrame(data)

class MatrixFactorizationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=10):
        super(MatrixFactorizationModel, self).__init__()
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
    def forward(self, user, item):
        # Get embeddings for users and items
        user_embedded = self.user_embedding(user)
        item_embedded = self.item_embedding(item)
        
        # Compute dot product of user and item embeddings
        return (user_embedded * item_embedded).sum(1)

# Instantiate model
num_users = df['user_id'].nunique()
num_items = df['item_id'].nunique()

model = MatrixFactorizationModel(num_users, num_items)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100

for epoch in range(epochs):
    model.train()

    # Forward pass
    predictions = model(users, items)

    # Compute loss
    loss = loss_fn(predictions, actual)