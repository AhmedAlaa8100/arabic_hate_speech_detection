# Arabic Hate Speech Detection using AraBERT

A complete deep learning project for detecting hate speech in Arabic text using the AraBERT model from Hugging Face Transformers.

## 🎯 Project Overview

This project implements a state-of-the-art Arabic hate speech detection system using:

- **Model**: `aubmindlab/bert-base-arabertv02` (AraBERT v2)
- **Dataset**: `manueltonneau/arabic-hate-speech-superset`
- **Framework**: PyTorch + Hugging Face Transformers
- **Task**: Binary classification (Hate Speech vs. Not Hate Speech)

## 📁 Project Structure

```
arabic_hate_speech_detection/
├── data/                          # Data storage (optional)
├── models/                        # Trained model checkpoints
├── results/                       # Training logs, metrics, and plots
├── src/                          # Source code
│   ├── __init__.py
│   ├── config.py                 # Configuration parameters
│   ├── dataset.py                # Data loading and preprocessing
│   ├── model.py                  # AraBERT model definition
│   ├── train.py                  # Training pipeline
│   ├── evaluate.py               # Evaluation pipeline
│   └── utils.py                  # Utility functions
├── main.py                       # Main entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd arabic_hate_speech_detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

```bash
# Train the model with default settings
python main.py --mode train

# Train with custom parameters
python main.py --mode train --batch-size 32 --learning-rate 3e-5 --epochs 5

# Train with frozen BERT (faster, less accurate)
python main.py --mode train --freeze-bert --dropout-rate 0.2
```

### 3. Evaluation

```bash
# Evaluate the trained model
python main.py --mode evaluate
```

### 4. Prediction

```bash
# Predict on a single text
python main.py --mode predict --text "هذا نص عربي للاختبار"

# Example predictions
python main.py --mode predict --text "أحب هذا المكان"
python main.py --mode predict --text "أنت شخص سيء جداً"
```

## ⚙️ Configuration

The project uses a centralized configuration system in `src/config.py`. Key parameters:

```python
# Model Configuration
model_name = "aubmindlab/bert-base-arabertv02"
max_length = 128
num_labels = 2

# Training Configuration
batch_size = 16
learning_rate = 2e-5
num_epochs = 3
warmup_steps = 100

# Data Configuration
dataset_name = "manueltonneau/arabic-hate-speech-superset"
validation_split = 0.1
```

## 📊 Features

### Data Processing

- **Arabic Text Normalization**: Handles various Arabic text variations
- **Automatic Dataset Loading**: Downloads and processes the Hugging Face dataset
- **Data Splitting**: Automatic train/validation/test splits
- **DataLoader Integration**: Efficient batching with PyTorch DataLoader

### Model Architecture

- **AraBERT v2**: Pre-trained Arabic BERT model
- **Custom Classification Head**: Dropout + Linear layer
- **Flexible Configuration**: Support for freezing BERT parameters
- **Model Checkpointing**: Automatic saving of best models

### Training Features

- **Progress Tracking**: Real-time training progress with tqdm
- **Learning Rate Scheduling**: Linear warmup + decay
- **Early Stopping**: Prevents overfitting
- **Gradient Clipping**: Stabilizes training
- **Comprehensive Logging**: Detailed training logs

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-Score**: Per-class and weighted metrics
- **Confusion Matrix**: Visual analysis of predictions
- **Confidence Analysis**: Prediction confidence statistics
- **Classification Report**: Detailed performance breakdown

### Visualization

- **Training Curves**: Loss and accuracy plots
- **Confusion Matrix**: Visual prediction analysis
- **Automatic Saving**: All plots saved to `results/` directory

## 🔧 Usage Examples

### Training with Custom Settings

```python
from src.config import Config
from src.train import Trainer
from src.dataset import DataProcessor

# Load configuration
config = Config()
config.batch_size = 32
config.learning_rate = 3e-5
config.num_epochs = 5

# Load data
data_processor = DataProcessor(config)
train_dataset, val_dataset, test_dataset = data_processor.load_dataset()
train_loader, val_loader, test_loader = data_processor.create_dataloaders(
    train_dataset, val_dataset, test_dataset
)

# Train model
trainer = Trainer(config)
results = trainer.train(train_loader, val_loader)
```

### Custom Prediction

```python
from src.model import ModelManager
from src.utils import clean_text
from transformers import AutoTokenizer

# Load model
config = Config()
model_manager = ModelManager(config)
model = model_manager.load_best_model()
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# Predict
text = "هذا نص للاختبار"
cleaned_text = clean_text(text)
encoding = tokenizer(cleaned_text, return_tensors='pt',
                   truncation=True, padding=True, max_length=128)

with torch.no_grad():
    outputs = model(encoding['input_ids'], encoding['attention_mask'])
    prediction = torch.argmax(outputs['logits'], dim=-1)
    confidence = torch.softmax(outputs['logits'], dim=-1)
```

## 📈 Results

The model typically achieves:

- **Accuracy**: 85-90% on the test set
- **F1-Score**: 0.85-0.90 (weighted)
- **Training Time**: ~30-60 minutes (depending on hardware)

Results are automatically saved to:

- `results/training.log`: Training logs
- `results/loss_curve.png`: Training curves
- `results/confusion_matrix.png`: Confusion matrix
- `results/metrics.json`: Detailed metrics
- `models/best_arabert_model.pt`: Best model checkpoint

## 🛠️ Advanced Usage

### Custom Dataset

To use your own dataset, modify the `load_dataset` method in `src/dataset.py`:

```python
def load_custom_dataset(self, data_path: str):
    # Load your custom dataset
    df = pd.read_csv(data_path)

    # Extract texts and labels
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # Create datasets
    train_dataset = ArabicHateSpeechDataset(texts, labels, self.tokenizer)
    return train_dataset
```

### Model Fine-tuning

For fine-tuning on specific domains:

```python
# Load pre-trained model
model = ArabicHateSpeechClassifier(
    model_name="aubmindlab/bert-base-arabertv02",
    num_labels=2,
    freeze_bert=False  # Allow fine-tuning
)

# Use lower learning rate for fine-tuning
config.learning_rate = 1e-5
```

### Hyperparameter Tuning

Key hyperparameters to tune:

- `learning_rate`: 1e-5 to 5e-5
- `batch_size`: 8, 16, 32, 64
- `max_length`: 64, 128, 256
- `dropout_rate`: 0.1, 0.2, 0.3
- `warmup_steps`: 50, 100, 200

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   ```bash
   # Reduce batch size
   python main.py --mode train --batch-size 8
   ```

2. **Dataset Download Issues**

   ```bash
   # Check internet connection and Hugging Face access
   # The dataset will be cached after first download
   ```

3. **Model Loading Errors**
   ```bash
   # Ensure the model is trained first
   python main.py --mode train
   python main.py --mode evaluate
   ```

### Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster training
2. **Batch Size**: Increase batch size if you have more GPU memory
3. **Mixed Precision**: Consider using `torch.cuda.amp` for faster training
4. **Data Parallel**: Use `DataParallel` for multi-GPU training

## 📚 Dependencies

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library
- **Datasets**: Hugging Face datasets library
- **Scikit-learn**: Machine learning utilities
- **Matplotlib/Seaborn**: Visualization
- **Pandas/NumPy**: Data manipulation
- **TQDM**: Progress bars

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **AraBERT Team**: For the excellent pre-trained Arabic BERT model
- **Hugging Face**: For the transformers and datasets libraries
- **Dataset Authors**: For providing the Arabic hate speech dataset

## 📞 Support

For questions or issues:

1. Check the troubleshooting section
2. Review the logs in `results/training.log`
3. Open an issue with detailed error information

---

**Happy Training! 🚀**
