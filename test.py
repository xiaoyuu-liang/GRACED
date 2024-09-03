import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import MLP

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # First layer
        self.convs.append(
            GINConv(
                MLP([input_dim, hidden_dim, hidden_dim]),
                train_eps=True
            )
        )
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GINConv(
                    MLP([hidden_dim, hidden_dim, hidden_dim]),
                    train_eps=True
                )
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Last layer
        self.convs.append(
            GINConv(
                MLP([hidden_dim, hidden_dim, output_dim]),
                train_eps=True
            )
        )
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        x = global_add_pool(x, batch)
        return x

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

def main():
    # Load the ENZYMES dataset
    path = 'data/ENZYMES'
    dataset = TUDataset(root=path, name='ENZYMES')

    # Split the dataset into training, validation, and test sets
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    train_dataset = dataset[:540]
    test_dataset = dataset[540:]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define the model
    input_dim = dataset.num_node_features
    hidden_dim = 64
    output_dim = dataset.num_classes
    num_layers = 5
    model = GIN(input_dim, hidden_dim, output_dim, num_layers)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {trainable_params}")

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, 201):
        loss = train(model, train_loader, optimizer, criterion)
        test_acc = test(model, test_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}')

if __name__ == '__main__':
    main()
