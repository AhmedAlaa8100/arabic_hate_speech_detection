# CUDA Usage Guide for Arabic Hate Speech Detection

## 🚀 Quick Start

Your system is **CUDA-ready**! Here's how to use CUDA effectively with your Arabic hate speech detection project.

## ✅ System Status

- **PyTorch Version**: 2.4.1+cu124
- **CUDA Available**: ✅ True
- **CUDA Version**: 12.4
- **Number of GPUs**: 1
- **GPU Name**: Available (check with `python -c "import torch; print(torch.cuda.get_device_name(0))"`)

## 🔧 Configuration Options

### 1. Device Configuration in `config.py`

```python
# Device Configuration
device: str = "auto"  # Options: "auto", "cuda", "cuda:0", "cpu"
force_cpu: bool = False  # Set to True to force CPU usage
mixed_precision: bool = True  # Enable mixed precision training
dataloader_pin_memory: bool = True  # Pin memory for faster GPU transfer
```

### 2. Device Selection Options

| Option     | Description                                | Use Case                         |
| ---------- | ------------------------------------------ | -------------------------------- |
| `"auto"`   | Automatically select best available device | **Recommended**                  |
| `"cuda"`   | Use first available GPU                    | When you want GPU specifically   |
| `"cuda:0"` | Use specific GPU (GPU 0)                   | Multi-GPU systems                |
| `"cpu"`    | Force CPU usage                            | Debugging or when GPU has issues |

## 🚀 Running with CUDA

### 1. Training with CUDA (Default)

```bash
# Automatic device selection (will use CUDA if available)
python main.py --mode train

# Explicitly use CUDA
python main.py --mode train --device cuda

# Use specific GPU
python main.py --mode train --device cuda:0
```

### 2. Training with Custom Settings

```bash
# Large batch size for better GPU utilization
python main.py --mode train --batch-size 32 --learning-rate 2e-5

# Mixed precision training (enabled by default)
python main.py --mode train --mixed-precision

# Force CPU usage (for debugging)
python main.py --mode train --force-cpu
```

### 3. Evaluation with CUDA

```bash
# Evaluate on GPU
python main.py --mode evaluate

# Evaluate on specific GPU
python main.py --mode evaluate --device cuda:0
```

### 4. Prediction with CUDA

```bash
# Predict on GPU
python main.py --mode predict --text "Your Arabic text here"
```

## ⚡ Performance Optimizations

### 1. Mixed Precision Training (Enabled by Default)

- **Benefits**: ~2x faster training, ~50% less memory usage
- **Requirements**: CUDA-compatible GPU (yours supports it!)
- **Status**: ✅ Enabled by default

### 2. Memory Optimizations

- **Pin Memory**: Enabled for faster CPU→GPU data transfer
- **Batch Size**: Adjust based on GPU memory
  - Start with 16, increase to 32 or 64 if you have more GPU memory
- **Gradient Accumulation**: For very large models

### 3. Recommended Settings for Your GPU

```python
# In config.py, adjust these for optimal performance:
batch_size: int = 32  # Increase from 16 for better GPU utilization
mixed_precision: bool = True  # Keep enabled
dataloader_pin_memory: bool = True  # Keep enabled
```

## 🔍 Monitoring GPU Usage

### 1. Check GPU Status

```bash
# Check if CUDA is working
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0))"

# Monitor GPU usage during training
nvidia-smi -l 1  # Updates every second
```

### 2. Memory Monitoring

```python
# Add this to your training script to monitor memory
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## 🛠️ Troubleshooting

### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:

```bash
# Reduce batch size
python main.py --mode train --batch-size 8

# Or reduce batch size in config.py
batch_size: int = 8
```

### 2. CUDA Not Available

**Error**: `CUDA not available`

**Solutions**:

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Performance Issues

**Slow Training**:

- Increase batch size (if memory allows)
- Enable mixed precision (already enabled)
- Check if data loading is the bottleneck

**Memory Issues**:

- Reduce batch size
- Disable mixed precision temporarily
- Use gradient checkpointing

## 📊 Expected Performance

With your CUDA setup, you should expect:

- **Training Speed**: ~2-5x faster than CPU
- **Memory Usage**: ~50% less with mixed precision
- **Batch Size**: Can handle 16-64 depending on model size
- **Training Time**: Significantly reduced for large datasets

## 🔧 Advanced Configuration

### 1. Multi-GPU Training (Future)

If you get multiple GPUs:

```python
# In config.py
device: str = "cuda:0"  # Use specific GPU
# Add DataParallel support in model.py
```

### 2. Custom Device Selection

```python
# In your code
from src.utils import get_device

# Get specific device
device = get_device("cuda:0", force_cpu=False)

# Force CPU for debugging
device = get_device("auto", force_cpu=True)
```

### 3. Environment Variables

```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run training
python main.py --mode train
```

## 📈 Monitoring and Logging

The system automatically logs:

- Device information at startup
- Mixed precision status
- Memory usage (if monitoring is enabled)
- Training speed improvements

## 🎯 Best Practices

1. **Always use `device="auto"`** for automatic device selection
2. **Keep mixed precision enabled** for better performance
3. **Monitor GPU memory** during training
4. **Start with smaller batch sizes** and increase gradually
5. **Use pin_memory=True** for faster data loading
6. **Check GPU utilization** with `nvidia-smi`

## 🚀 Quick Commands

```bash
# Check CUDA status
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0))"

# Train with optimal settings
python main.py --mode train --batch-size 32

# Evaluate model
python main.py --mode evaluate

# Predict text
python main.py --mode predict --text "النص العربي هنا"
```

Your system is ready for high-performance CUDA training! 🎉
