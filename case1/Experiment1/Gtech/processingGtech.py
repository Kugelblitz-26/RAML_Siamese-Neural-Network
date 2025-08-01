# Data preprocessing and augmentation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset for LFW
class LFWDataset(Dataset):
    def __init__(self, root_dir, pairs_file=None, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # Get all person directories
        self.people = [d for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d))]
        
        # Create image paths and labels
        self.image_paths = []
        self.labels = []
        
        for person_idx, person in enumerate(self.people):
            person_dir = os.path.join(root_dir, person)
            images = [f for f in os.listdir(person_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img in images:
                self.image_paths.append(os.path.join(person_dir, img))
                self.labels.append(person_idx)
        
        self.num_classes = len(self.people)
        print(f"Dataset loaded: {len(self.image_paths)} images, {self.num_classes} people")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image and label
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, 0

# Siamese Dataset for pairs
class SiameseDataset(Dataset):
    def __init__(self, base_dataset, num_pairs=10000):
        self.base_dataset = base_dataset
        self.num_pairs = num_pairs
        
        # Group images by label
        self.label_to_indices = {}
        for idx, (_, label) in enumerate(base_dataset):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        # Generate pairs
        self.pairs = []
        self.labels = []
        
        # Generate positive pairs (same person)
        for _ in range(num_pairs // 2):
            label = np.random.choice(list(self.label_to_indices.keys()))
            if len(self.label_to_indices[label]) >= 2:
                idx1, idx2 = np.random.choice(self.label_to_indices[label], 2, replace=False)
                self.pairs.append((idx1, idx2))
                self.labels.append(1)  # Same person
        
        # Generate negative pairs (different people)
        for _ in range(num_pairs // 2):
            label1, label2 = np.random.choice(list(self.label_to_indices.keys()), 2, replace=False)
            idx1 = np.random.choice(self.label_to_indices[label1])
            idx2 = np.random.choice(self.label_to_indices[label2])
            self.pairs.append((idx1, idx2))
            self.labels.append(0)  # Different people
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        label = self.labels[idx]
        
        img1, _ = self.base_dataset[idx1]
        img2, _ = self.base_dataset[idx2]
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)