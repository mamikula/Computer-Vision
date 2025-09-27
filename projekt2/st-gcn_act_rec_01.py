import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Step 1: Generate Toy Data for Human Skeleton Motion
num_frames = 100  # Number of frames in the sequence
num_joints = 7  # Number of joints in the skeleton
num_features = 2  # (x, y) positions for each joint
num_classes = 3  # 3 classes for human actions (e.g., walk, run, jump)

# Skeleton structure: 7 joints with edges connecting adjacent joints
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
                           [1, 2, 3, 4, 5, 6, 0]], dtype=torch.long)  # Connecting each joint to the next

# Simulating motion data for 100 frames. Randomly generate (x, y) positions for each joint
x_data = torch.randn(num_frames, num_joints, num_features)

# Generate toy labels for 3 classes (e.g., walk, run, jump)
y_data = torch.tensor([i % num_classes for i in range(num_frames)], dtype=torch.long)

# Step 2: Create a Data object for each frame (graph data for each time step)
data_frames = []
for t in range(num_frames):
    data_frame = Data(x=x_data[t], edge_index=edge_index, y=y_data[t].unsqueeze(0))
    data_frames.append(data_frame)

# Step 3: Define the ST-GCN Model
class STGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints, num_frames):
        super(STGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)  # First graph convolution layer
        self.conv2 = GCNConv(64, 128)  # Second graph convolution layer
        self.temporal_conv = nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0))  # Temporal convolution
        self.fc = nn.Linear(128 * num_frames * num_joints, out_channels)  # Fully connected layer for classification

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Graph convolution layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Reshape x to include the time dimension (num_frames)
        # We will repeat the graph features across time frames
        x = x.unsqueeze(0)  # Add batch dimension: shape becomes [1, num_joints, 128]
        x = x.permute(0, 2, 1)  # Change shape to [1, 128, num_joints] (channels, joints)
        x = x.repeat(1, 1, num_frames)  # Repeat across time frames: [1, 128, num_joints, num_frames]

        # Apply the temporal convolution (across time)
        x = x.unsqueeze(3)  # Add time_steps dimension: [1, 128, num_joints, num_frames, 1]
        x = F.relu(self.temporal_conv(x))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten all but the batch dimension
        
        # Output classification
        x = self.fc(x)
        return x

# Step 4: Training the ST-GCN

# Initialize the model, optimizer, and loss function
model = STGCN(in_channels=num_features, out_channels=num_classes, num_joints=num_joints, num_frames=num_frames)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Function to train the model
# def train(model, data_frames, optimizer, criterion, epochs=200):
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         # Concatenate data frames along time dimension
#         out = model(data_frames[0])  # Only passing one frame for simplicity here (modify for sequence)
#
#         # Compute loss and backpropagate
#         loss = criterion(out, data_frames[0].y)
#         loss.backward()
#         optimizer.step()
#
#         if epoch % 20 == 0:
#             print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def train(model, data_frames, optimizer, criterion, epochs=200):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Concatenate features from all frames into a single tensor
        x_seq = torch.stack([data.x for data in data_frames])  # Shape: [num_frames, num_joints, num_features]
        edge_index = data_frames[0].edge_index  # Edge index is the same for all frames
        y_seq = torch.tensor([data.y.item() for data in data_frames], dtype=torch.long)  # Labels for all frames

        # Create a single Data object for the sequence
        sequence_data = Data(x=x_seq, edge_index=edge_index, y=y_seq)

        # Pass the sequence through the model
        out = model(sequence_data)  # Model processes the full sequence

        # Compute loss and backpropagate
        loss = criterion(out, sequence_data.y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')


# Step 5: Evaluation

def test(model, data_frame):
    model.eval()
    with torch.no_grad():
        out = model(data_frame)  # Run the forward pass
        pred = out.argmax(dim=1)  # Get the predicted class
        correct = (pred == data_frame.y).sum()  # Compare with true label
        accuracy = correct / len(data_frame.y)
        return accuracy.item()

# Train the model on the toy data
train(model, data_frames, optimizer, criterion, epochs=200)

# Test the model on the first frame (toy data)
accuracy = test(model, data_frames[0])
print(f'Accuracy on test data: {accuracy * 100:.2f}%')

# Optional: Visualize the graph for the first frame
G = nx.Graph()
for i in range(num_joints):
    G.add_node(i)
for edge in edge_index.T:
    G.add_edge(edge[0].item(), edge[1].item())

# Visualizing the skeleton graph
plt.figure(figsize=(6, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=16)
plt.title('Skeleton Graph (Frame 1)')
plt.show()




def load_data_from_folders(folder_paths, num_features=2, num_frames=150, edge_index=None):
    """
    Load data from CSV files in the given folders and return a list of DataFrames for PyTorch Geometric.

    Arguments:
    folder_paths -- list of folder paths containing the CSV files
    num_features -- number of features (coordinates per joint)
    num_frames -- number of frames to process
    edge_index -- tensor defining the skeleton connections (edges between joints)

    Returns:
    data_frames -- list of Data objects for each frame in the sequence
    """
    x_data_list = []
    y_data_list = []

    # Loop through each folder and file
    for folder_index, folder_path in enumerate(folder_paths, start=1):
        for seq_num in range(1, 51):  # seq_001 to seq_050
            file_name = f"seq_{seq_num:03d}.csv"  # Format the file name (seq_001.csv)
            file_path = os.path.join(folder_path, file_name)
            # Check if the file exists
            if os.path.exists(file_path):
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path, skiprows=2, header=None)

                # Process the data into x_data
                for i in range(len(df)):
                    frame_data = df.iloc[i].dropna().values.reshape(-1, num_features)
                    x_data_list.append(frame_data)
                    y_data_list.append(folder_index)  # Assign label based on folder (1, 2, or 3)

    # Convert x_data and y_data to tensors
    x_data = torch.tensor(x_data_list, dtype=torch.float32)
    y_data = torch.tensor(y_data_list, dtype=torch.long)
    data_frames = []
    for t in range(len(y_data)):
        data_frame = Data(x=x_data[t], edge_index=edge_index, y=y_data[t].unsqueeze(0))
        data_frames.append(data_frame)
    return data_frames