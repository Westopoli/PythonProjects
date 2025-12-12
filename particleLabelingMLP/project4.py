import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Define the MLP model for binary classification
class BinaryClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=64):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def train_mlp(X, y):
    """Train a single MLP model"""
    # Standardize the input features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / (X_std + 1e-8)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_normalized)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    # Initialize model
    model = BinaryClassifier(input_size=2, hidden_size=64)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0025)
    
    # Training loop
    num_epochs = 5000
    batch_size = 64
    
    for epoch in range(num_epochs):
        indices = np.random.permutation(len(X_tensor))
        
        for i in range(0, len(X_tensor), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_tensor[batch_indices]
            y_batch = y_tensor[batch_indices]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    return model, X_mean, X_std

def train_model(X, y):
    """Train both SVM and MLP, return both for ensemble"""
    
    # Train SVM with RBF kernel (excellent for circular boundaries)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try multiple C values and pick best based on training accuracy
    best_svm = None
    best_score = 0
    
    for C in [0.1, 1.0, 10.0, 100.0]:
        svm = SVC(kernel='rbf', C=C, gamma='scale', probability=True)
        svm.fit(X_scaled, y)
        score = svm.score(X_scaled, y)
        if score > best_score:
            best_score = score
            best_svm = svm
    
    # Train MLP
    mlp_model, X_mean, X_std = train_mlp(X, y)
    
    return {
        'svm': best_svm,
        'scaler': scaler,
        'mlp': mlp_model,
        'X_mean': X_mean,
        'X_std': X_std
    }

def run_model(model_dict, X):
    """Ensemble prediction: average SVM and MLP predictions"""
    
    # Get SVM predictions (probabilities)
    X_scaled = model_dict['scaler'].transform(X)
    svm_probs = model_dict['svm'].predict_proba(X_scaled)[:, 1]  # Probability of class 1
    
    # Get MLP predictions (probabilities)
    X_normalized = (X - model_dict['X_mean']) / (model_dict['X_std'] + 1e-8)
    X_tensor = torch.FloatTensor(X_normalized)
    
    mlp = model_dict['mlp']
    mlp.eval()
    with torch.no_grad():
        mlp_probs = mlp(X_tensor).squeeze().numpy()
    
    # Average the probabilities
    avg_probs = (svm_probs + mlp_probs) / 2.0
    
    # Convert to binary predictions
    predictions = (avg_probs >= 0.5).astype(np.int32)
    
    return predictions

# Main execution
X_train = np.load('data_train.npy')
y_train = np.load('label_train.npy')

model = train_model(X_train, y_train)

X_test = np.load('data_test.npy')
y_test = run_model(model, X_test)

assert y_test.shape == (len(X_test),) and y_test.dtype == np.int32

for y in y_test:
    print(y)