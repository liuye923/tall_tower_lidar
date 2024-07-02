import torch
import numpy as np
import json
import logging
from sklearn.metrics import classification_report
from models.transformer_encoder import TransformerEncoder
from models.projection_head import ProjectionHead
from models.linear_classifier import LinearClassifier
from models.utils import load_model

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def evaluate_model(data, labels, config):
    input_dim = data.shape[2]
    num_classes = len(np.unique(labels))

    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device") == "cuda" else "cpu")
    
    transformer_encoder = TransformerEncoder(input_dim=input_dim, num_layers=6, num_heads=8, hidden_dim=512).to(device)
    projection_head = ProjectionHead(input_dim=512, projection_dim=128).to(device)
    linear_classifier = LinearClassifier(input_dim=128, num_classes=num_classes).to(device)

    transformer_encoder = load_model(transformer_encoder, config["model_paths"]["transformer_encoder"])
    projection_head = load_model(projection_head, config["model_paths"]["projection_head"])
    linear_classifier = load_model(linear_classifier, config["model_paths"]["linear_classifier"])

    transformer_encoder.eval()
    projection_head.eval()
    linear_classifier.eval()

    predictions = []
    with torch.no_grad():
        for i in range(len(data)):
            sample = torch.tensor(data[i], dtype=torch.float32).unsqueeze(0).to(device)
            encoded_features = transformer_encoder(sample)
            projected_features = projection_head(encoded_features)
            outputs = linear_classifier(projected_features)
            predicted_class = torch.argmax(outputs, dim=1).item()
            predictions.append(predicted_class)

    print(classification_report(labels, predictions))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config_path = "config.json"
    config = load_config(config_path)

    try:
        data = np.load(config["data_path"])
        labels = np.load(config["labels_path"])
        
        if data.ndim != 3:
            raise ValueError("Data should be a 3D array")
        if labels.ndim != 1:
            raise ValueError("Labels should be a 1D array")

        evaluate_model(data, labels, config)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
