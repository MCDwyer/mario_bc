import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score
import matplotlib.pyplot as plt

FILEPATH = "/Users/mdwyer/Documents/Code/PhD_Mario_Work/mario_bc/bc_datasets/"

# --- 1. Custom Dataset from NumPy arrays ---
class StateActionDatasetFromNumpy(Dataset):
    def __init__(self, images_np, actions_np):
        self.images = torch.tensor(images_np, dtype=torch.float32).unsqueeze(1)  # shape: [N, 1, 84, 84]
        self.actions = torch.tensor(actions_np, dtype=torch.long)

        # Normalize images to [-1, 1]
        self.images = (self.images / 127.5) - 1.0

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.images[idx], self.actions[idx]

# --- 2. CNN Classifier (same as before) ---
class CNNClassifier(nn.Module):
    def __init__(self, num_actions=13):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2),  # [1,84,84] → [32,40,40]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), # → [64,19,19]
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 19 * 19, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# --- 3. Training Loop ---
def train_model(model, dataloader, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# --- 4. Anomaly Detection Function ---
def is_anomalous(model, image_np, actual_action, threshold=0.01):
    model.eval()
    with torch.no_grad():
        image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,84,84]
        image_tensor = (image_tensor / 127.5) - 1.0  # normalize
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        prob_of_actual = probs[0, actual_action].item()
        return prob_of_actual < threshold, prob_of_actual

def load_data(training_data_name, level):

    full_filepath = f"{FILEPATH}{level}_{training_data_name}.pkl"

    with open(full_filepath, 'rb') as file:
        loaded_data = pickle.load(file)

    actions, observations = zip(*loaded_data)

    print(f"Loading {level} demo data from {full_filepath}.")

    observations = np.array(observations)
    # print(observations.shape)
    observations = observations.reshape(observations.shape[0], observations.shape[1], observations.shape[2])
    print(observations.shape)

    return observations, actions

def save_data(states, actions, level, filename):
    if states.shape[0] != actions.shape[0]:
        print(f"Something has gone wrong... :(")

    combined = zip(actions, states)

    filepath = f"{FILEPATH}{level}_{filename}"
    with open(f"{filepath}.pkl", "wb") as file:
        pickle.dump(combined, file)
        print(f"Dataset saved to: {filepath}.pkl")

def get_scores(model, images_np, actions_np):
    scores = []
    for img, action in zip(images_np, actions_np):
        _, confidence = is_anomalous(model, img, action, threshold=0.0)  # get score only
        scores.append(confidence)
    return np.array(scores)


def find_anomalous_data(level):
    # Assuming you already have:
    # images_np: shape [N, 84, 84], dtype uint8 or float32
    # actions_np: shape [N], integers in [0, 12]

    observations, actions = load_data("expert_distance", level)

    X_train, X_test, y_train, y_test = train_test_split(observations, actions, test_size=0.1, random_state=42)

    train_dataset = StateActionDatasetFromNumpy(X_train, y_train)

    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = CNNClassifier()
    train_model(model, dataloader, epochs=5)

    # Assuming you have:
    # - val_images_np: [M, 84, 84]
    # - val_actions_np: [M]

    scores = get_scores(model, X_test, y_test)

    for percentage in [0.01, 0.05, 0.1, 0.5, 1]:
        print(percentage)
        print(np.percentile(scores, percentage))

    all_observations, all_actions = load_data("amalgam", level)

    expert_obs = []
    expert_actions = []

    non_expert_obs = []
    non_expert_actions = []
    
    anomalous = 0
    scores = []


    for observation, action in zip(all_observations, all_actions):
        anomaly, confidence = is_anomalous(model, observation, action)
        scores.append(confidence)
        if anomaly:
            non_expert_obs.append(observation)
            non_expert_actions.append(action)
            # print("Anomalous?", anomaly, "Confidence:", confidence)
            anomalous += 1
        else:
            expert_obs.append(observation)
            expert_actions.append(action)

    print("all data percentage")
    for percentage in [0.01, 0.05, 0.1, 0.5, 1]:
        print(percentage)
        print(np.percentile(scores, percentage))

    non_expert_obs = np.array(non_expert_obs)
    # non_expert_obs = non_expert_obs.unsqueeze()
    non_expert_obs = non_expert_obs.reshape(non_expert_obs.shape[0], non_expert_obs.shape[1], non_expert_obs.shape[2], 1)
    non_expert_actions = np.array(non_expert_actions)    
    
    print(non_expert_obs.shape)
    print(non_expert_actions.shape)

    expert_obs = np.array(expert_obs)
    # expert_obs = expert_obs.unsqueeze()
    expert_obs = expert_obs.reshape(expert_obs.shape[0], expert_obs.shape[1], expert_obs.shape[2], 1)
    expert_actions = np.array(expert_actions)

    print(expert_obs.shape)
    print(expert_actions.shape)

    print(f"{anomalous} Anomalous State-Action Pairs, compared with {len(actions)} Expert State-Action Pairs")
    save_data(expert_obs, expert_actions, level, "expert_classifier")
    save_data(non_expert_obs, non_expert_actions, level, "nonexpert_classifier")

# find_anomalous_data("Level1-1")

for level in ["Level1-1", "Level2-1", "Level3-1", "Level4-1", "Level5-1", "Level6-1", "Level7-1", "Level8-1"]:
    print(level)
    find_anomalous_data(level)
