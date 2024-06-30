import torch
import numpy as np
from sklearn.metrics import classification_report
from models.transformer_encoder import TransformerEncoder
from models.projection_head import ProjectionHead
from models.linear_classifier import LinearClassifier
from models.utils import load_model

def evaluate_model(data, labels):
    input_dim = data.shape[2]
    num_classes = len(np.unique(labels))

    transformer_encoder = TransformerEncoder(input_dim=input_dim, num_layers=6, num_heads=8, hidden_dim=512)
    projection_head = ProjectionHead(input_dim=512, projection_dim=128)
    linear_classifier = LinearClassifier(input_dim=128, num_classes=num_classes)

    transformer_encoder = load_model(transformer_encoder, 'models/transformer_encoder.pth')
    projection_head = load_model(projection_head, 'models/projection_head.pth')
    linear_classifier = load_model(linear_classifier, 'models/linear_classifier.pth')

    transformer_encoder.eval()
    projection_head.eval()
    linear_classifier.eval()

    predictions = []
    with torch.no_grad():
        for i in range(len(data)):
            sample = torch.tensor(data[i], dtype=torch.float32).unsqueeze(0)
            encoded_features = transformer_encoder(sample)
            projected_features = projection_head(encoded_features)
            outputs = linear_classifier(projected_features)
            predicted_class = torch.argmax(outputs, dim=1).item()
            predictions.append(predicted_class)

    print(classification_report(labels, predictions))

if __name__ == "__main__":
    data_path = "data/processed/preprocessed_data.npy"
    labels_path = "data/processed/labels.npy"
    data = np.load(data_path)
    labels = np.load(labels_path)
    evaluate_model(data, labels)
