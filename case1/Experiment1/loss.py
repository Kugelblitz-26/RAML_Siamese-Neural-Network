# Loss Functions

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embedding1, embedding2, label):
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                         label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))
        return loss

class LogisticLoss(nn.Module):
    def __init__(self):
        super(LogisticLoss, self).__init__()
        
    def forward(self, embedding1, embedding2, label):
        # Compute similarity score
        similarity = F.cosine_similarity(embedding1, embedding2)
        # Convert to probability
        prob = torch.sigmoid(similarity * 10)  # Scale for better gradients
        # Binary cross entropy
        loss = F.binary_cross_entropy(prob, label)
        return loss

# Training Functions

def train_contrastive(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (img1, img2, label) in enumerate(dataloader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        optimizer.zero_grad()
        embedding1, embedding2 = model(img1, img2)
        loss = criterion(embedding1, embedding2, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)

def train_triplet(model, base_dataset, criterion, optimizer, device, batch_size=32):
    model.train()
    total_loss = 0
    num_batches = 100  # Number of triplet batches per epoch
    
    # Group samples by label
    label_to_indices = {}
    for idx in range(len(base_dataset)):
        _, label = base_dataset[idx]
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    labels_with_multiple = [label for label, indices in label_to_indices.items() 
                           if len(indices) >= 2]
    
    for batch_idx in range(num_batches):
        anchors, positives, negatives = [], [], []
        
        for _ in range(batch_size):
            # Select anchor label (must have at least 2 samples)
            anchor_label = np.random.choice(labels_with_multiple)
            anchor_idx, positive_idx = np.random.choice(
                label_to_indices[anchor_label], 2, replace=False)
            
            # Select negative label
            negative_label = np.random.choice(
                [l for l in labels_with_multiple if l != anchor_label])
            negative_idx = np.random.choice(label_to_indices[negative_label])
            
            anchor_img, _ = base_dataset[anchor_idx]
            positive_img, _ = base_dataset[positive_idx]
            negative_img, _ = base_dataset[negative_idx]
            
            anchors.append(anchor_img)
            positives.append(positive_img)
            negatives.append(negative_img)
        
        anchors = torch.stack(anchors).to(device)
        positives = torch.stack(positives).to(device)
        negatives = torch.stack(negatives).to(device)
        
        optimizer.zero_grad()
        anchor_emb = model.get_embedding(anchors)
        positive_emb = model.get_embedding(positives)
        negative_emb = model.get_embedding(negatives)
        
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 20 == 0:
            print(f'Triplet Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches