import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from data_preprocessing import create_data_loaders
from models import StudentNetwork
from config import Config
import os

def calculate_metrics(confusion_mat):
    # Calculate weighted accuracy (WA)
    total_samples = np.sum(confusion_mat)
    correct_predictions = np.sum(np.diag(confusion_mat))
    weighted_accuracy = correct_predictions / total_samples
    
    # Calculate unweighted accuracy (UA)
    class_accuracies = np.diag(confusion_mat) / np.sum(confusion_mat, axis=1)
    unweighted_accuracy = np.mean(class_accuracies)
    
    return weighted_accuracy, unweighted_accuracy

def evaluate_model(model_path):
    # Load model
    model = StudentNetwork().to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load data
    _, test_loader = create_data_loaders(Config.DATA_DIR)
    
    # Initialize confusion matrix
    num_classes = len(Config.EMOTION_LABELS)
    confusion_mat = np.zeros((num_classes, num_classes))
    
    # Evaluate
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_mat[t.long(), p.long()] += 1
    
    # Calculate metrics
    weighted_acc, unweighted_acc = calculate_metrics(confusion_mat)
    
    # Print results
    print("\nConfusion Matrix:")
    print(confusion_mat)
    print("\nClass-wise Accuracy:")
    for i, emotion in enumerate(Config.EMOTION_LABELS):
        accuracy = confusion_mat[i, i] / np.sum(confusion_mat[i])
        print(f"{emotion}: {accuracy:.2%}")
    
    print(f"\nWeighted Accuracy (WA): {weighted_acc:.2%}")
    print(f"Unweighted Accuracy (UA): {unweighted_acc:.2%}")

if __name__ == '__main__':
    model_path = os.path.join(Config.MODEL_SAVE_DIR, 'student_best.pth')
    evaluate_model(model_path) 