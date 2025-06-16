# Training file
# src/train.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SiameseNetwork
from sampling import random_sampling, hard_negative_sampling, semi_hard_negative_sampling, distance_weighted_sampling


ndef contrastive_loss(out1, out2, label, margin=1.0):
    """Contrastive loss for pairs."""
    euclidean = F.pairwise_distance(out1, out2)
    loss = (1 - label) * 0.5 * euclidean.pow(2) + label * 0.5 * torch.clamp(margin - euclidean, min=0).pow(2)
    return loss.mean()


def train(args):
    # Data
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    if args.dataset == 'mnist':
        dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    elif args.dataset == 'omniglot':
        dataset = datasets.Omniglot(root='data', background=True, download=True, transform=transform)
    else:
        raise ValueError('Unknown dataset')

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork(embedding_dim=args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            # sample pairs/triplets indices
            neg_idx = None
            if args.strategy == 'random':
                neg_idx = random_sampling(labels)
            elif args.strategy == 'hard_negative':
                out, _ = model(data, data)
                neg_idx = hard_negative_sampling(out, labels)
            elif args.strategy == 'semi_hard':
                out, _ = model(data, data)
                neg_idx = semi_hard_negative_sampling(out, labels, margin=args.margin)
            elif args.strategy == 'distance_weighted':
                out, _ = model(data, data)
                neg_idx = distance_weighted_sampling(out, labels)
            else:
                raise ValueError('Invalid strategy')

            # form pairs
            anchor, positive = data, data
            negative = data[neg_idx]

            out_anchor, out_positive = model(anchor, positive)
            _, out_negative = model(anchor, negative)

            # compute losses
            # here using triplet loss as example
            triplet_loss = nn.TripletMarginLoss(margin=args.margin)(out_anchor, out_positive, out_negative)
            loss = triplet_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch}/{args.epochs}, Loss: {epoch_loss/len(loader):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, required=True,
                        choices=['random', 'hard_negative', 'semi_hard', 'distance_weighted'])
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mnist', 'omniglot'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--embedding_dim', type=int, default=128)
    args = parser.parse_args()
    train(args)
