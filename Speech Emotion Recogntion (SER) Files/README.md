# Emotion Recognition using Vision Transformers with Knowledge Distillation

This project implements an emotion recognition system using Vision Transformers (ViT) with knowledge distillation, trained on the CREMA-D dataset. The system consists of a Teacher network and a Student network, where the Student learns from the Teacher through a three-stage training process.

## Project Structure

```
.
├── data/
│   └── CREMA-D/          # Place CREMA-D dataset here
├── src/
│   ├── config.py         # Configuration parameters
│   ├── data_preprocessing.py  # Data loading and preprocessing
│   ├── models.py         # Teacher and Student network implementations
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── models/               # Directory for saved models
├── runs/                 # TensorBoard logs
└── requirements.txt      # Project dependencies
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and prepare the CREMA-D dataset:
- Download the CREMA-D dataset from [here](https://github.com/CheyneyComputerScience/CREMA-D)
- Extract the dataset into the `data/CREMA-D` directory
- The directory structure should be:
```
data/
└── CREMA-D/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    └── sad/
```

## Training

To train the model, run:
```bash
python src/train.py
```

The training process consists of three stages:
1. Teacher Network Training
2. Student Network Feature Matching
3. Student Network Final Training

Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir runs
```

## Evaluation

To evaluate the trained model, run:
```bash
python src/evaluate.py
```

This will calculate and display:
- Confusion matrix
- Class-wise accuracy
- Weighted Accuracy (WA)
- Unweighted Accuracy (UA)

## Configuration

All hyperparameters and settings can be modified in `src/config.py`, including:
- Audio processing parameters
- Model architecture parameters
- Training parameters
- Data split ratio
- Device configuration (CUDA/MPS/CPU)

## Notes

- The code automatically uses CUDA if available, falls back to MPS (Apple Silicon), and finally CPU
- Training logs are saved in the `runs` directory
- Best model checkpoints are saved in the `models` directory
- Data augmentation is applied during training (noise, time stretching, pitch shifting)
