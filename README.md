# Arabic Hate Speech Detection

A complete deep learning project for detecting hate speech in Arabic text using AraBERT (Arabic BERT). This project implements state-of-the-art natural language processing techniques to classify Arabic text as either hate speech or not.

## 🚀 Features

- **AraBERT Integration**: Uses the powerful `aubmindlab/bert-base-arabertv02` model
- **Comprehensive Training Pipeline**: Complete training, validation, and evaluation system
- **Multiple Loss Functions**: Support for Cross-Entropy, Weighted, and Focal Loss
- **Advanced Text Preprocessing**: Specialized Arabic text cleaning and normalization
- **Flexible Configuration**: YAML-based configuration system
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and analysis
- **Mixed Precision Training**: Optimized for modern GPUs
- **Well-Structured Codebase**: Following Python best practices

## 📁 Project Structure

```
arabic_hate_speech_detection/
├── README.md
├── setup.py
├── requirements.txt
├── pyproject.toml
├── main.py
├── config/
│   ├── __init__.py
│   ├── config.yaml
│   └── default_config.yaml
├── src/
│   └── arabic_hate_speech/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── model.py
│       │   └── losses.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   └── preprocessing.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   └── callbacks.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── evaluator.py
│       │   └── metrics.py
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           ├── device.py
│           └── helpers.py
├── scripts/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   ├── build_dataset.py
│   └── resume_training.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── combined/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── models/
│   ├── .gitkeep
│   └── checkpoints/
├── results/
│   ├── .gitkeep
│   ├── logs/
│   ├── plots/
│   └── metrics/
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_dataset.py
│   └── test_utils.py
├── notebooks/
│   └── .gitkeep
└── docs/
    ├── .gitkeep
    └── api/
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/arabic-hate-speech-detection.git
cd arabic-hate-speech-detection

# Install the package in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "from arabic_hate_speech import Config; print('✅ Installation successful!')"
```

## 🚀 Quick Start

### 1. Training

```bash
# Train with default configuration
python main.py --mode train

# Train with custom parameters
python main.py --mode train --epochs 5 --batch-size 32 --learning-rate 3e-5

# Train with frozen BERT
python main.py --mode train --freeze-bert
```

### 2. Evaluation

```bash
# Evaluate the trained model
python main.py --mode evaluate
```

### 3. Prediction

```bash
# Predict on a single text
python main.py --mode predict --text "هذا نص عربي للاختبار"
```

## ⚙️ Configuration

The project uses YAML configuration files for easy customization:

### Main Configuration (`config/config.yaml`)

```yaml
# Model Configuration
model:
  name: "aubmindlab/bert-base-arabertv02"
  num_labels: 2
  max_length: 128
  dropout_rate: 0.1

# Training Configuration
training:
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 3
  early_stopping_patience: 3

# Data Configuration
data:
  dataset_name: "manueltonneau/arabic-hate-speech-superset"
  custom_dataset: false
  validation_split: 0.1
```

### Command Line Overrides

You can override any configuration parameter via command line:

```bash
python main.py --mode train --batch-size 32 --learning-rate 3e-5 --epochs 5
```

## 📊 Usage Examples

### Programmatic Usage

```python
from arabic_hate_speech import Config, DataProcessor, Trainer, Evaluator

# Load configuration
config = Config.from_json()

# Load and prepare data
data_processor = DataProcessor(config)
train_dataset, val_dataset, test_dataset = data_processor.load_dataset()
train_loader, val_loader, test_loader = data_processor.create_dataloaders(
    train_dataset, val_dataset, test_dataset
)

# Train model
trainer = Trainer(config)
results = trainer.train(train_loader, val_loader)

# Evaluate model
evaluator = Evaluator(config)
evaluation_results = evaluator.evaluate_and_save_results(test_loader)
```

### Custom Dataset

```python
# Enable custom dataset in config
config.custom_dataset = True
config.custom_dataset_path = "path/to/your/dataset"

# Your dataset should have the following structure:
# data/
#   ├── train.csv (columns: text, label)
#   ├── val.csv (columns: text, label)
#   └── test.csv (columns: text, label)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=arabic_hate_speech

# Run specific test
pytest tests/test_model.py
```

## 📈 Performance

The model achieves the following performance on the Arabic Hate Speech dataset:

- **Accuracy**: 87.21%
- **F1-Score**: 0.85
- **Precision**: 0.84
- **Recall**: 0.86

## 🔧 Advanced Features

### Mixed Precision Training

Enable mixed precision for faster training on modern GPUs:

```yaml
training:
  mixed_precision: true
```

### Class Imbalance Handling

```yaml
# Use weighted loss for imbalanced datasets
loss:
  function: "weighted"

# Or use focal loss
loss:
  function: "focal"
  focal_alpha: 1.0
  focal_gamma: 2.0
```

### Custom Text Preprocessing

```python
from arabic_hate_speech.data.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor(
    remove_diacritics=True,
    remove_punctuation=False,
    normalize_whitespace=True,
    remove_urls=True
)

cleaned_text = preprocessor.clean_text("نص عربي للتنظيف")
```

## 📝 Logging

The project includes comprehensive logging:

- **Training logs**: `results/logs/training.log`
- **Evaluation logs**: `results/logs/evaluation.log`
- **Model logs**: `results/logs/model.log`

## 🎯 Model Architecture

The model uses a BERT-based architecture:

1. **AraBERT Encoder**: Pre-trained Arabic BERT model
2. **Dropout Layer**: Regularization
3. **Classification Head**: Linear layer for binary classification
4. **Loss Function**: Configurable (CE, Weighted, Focal)

## 📊 Results and Visualizations

The project automatically generates:

- **Training curves**: Loss and accuracy plots
- **Confusion matrix**: Classification performance
- **Metrics**: Comprehensive evaluation metrics
- **Analysis**: Detailed prediction analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [AraBERT](https://github.com/aub-mindlab/arabert) for the pre-trained Arabic BERT model
- [Hugging Face](https://huggingface.co/) for the transformers library
- [Arabic Hate Speech Dataset](https://huggingface.co/datasets/manueltonneau/arabic-hate-speech-superset) for the training data

## 📞 Support

If you have any questions or need help, please:

1. Check the [Issues](https://github.com/your-username/arabic-hate-speech-detection/issues) page
2. Create a new issue if your problem isn't already reported
3. Contact the maintainers

## 🔮 Future Work

- [ ] Support for more Arabic dialects
- [ ] Real-time inference API
- [ ] Model quantization for deployment
- [ ] Multi-label classification
- [ ] Explainable AI features

---

**Made with ❤️ for the Arabic NLP community**
