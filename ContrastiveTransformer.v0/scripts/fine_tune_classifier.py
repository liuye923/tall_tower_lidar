import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from models.transformer_encoder import TransformerEncoder
from models.projection_head import ProjectionHead
from models.linear_classifier import LinearClassifier
from models.utils import load_model

def fine_tune_classifier(data, labels, num_epochs=50, batch_size=32, learning_rate=0.001):
    input_dim = data.shape[2]
    num_classes = len(np.unique(labels))

    transformer_encoder = TransformerEncoder(input_dim=input_dim, num_layers=6, num_heads=8, hidden_dim=512)
    projection_head = ProjectionHead(input_dim=512, projection_dim=128)
    linear_classifier = LinearClassifier(input_dim=128, num_classes=num_classes)

    transformer_encoder = load_model(transformer_encoder, 'models/transformer_encoder.pth')
    projection_head = load_model(projection_head, 'models/projection_head.pth')

    optimizer = optim.Adam(linear_classifier.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i in range(0, len(data), batch_size):
            batch_data = torch.tensor(data[i:i+batch_size], dtype=torch.float32)
            batch_labels = torch.tensor(labels[i:i+batch_size], dtype=torch.long)

            encoded_features = transformer_encoder(batch_data)
            projected_features = projection_head(encoded_features)
            outputs = linear_classifier(projected_features)

            loss = F.cross_entropy(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(linear_classifier.state_dict(), 'models/linear_classifier.pth')

if __name__ == "__main__":
    data_path = "data/processed/preprocessed_data.npy"
    labels_path = "data/processed/labels.npy"
    data = np.load(data_path)
    labels = np.load(labels_path)
    fine_tune_classifier(data, labels)
