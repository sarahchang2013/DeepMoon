# PyTorch Migration Notes

## Overview
This document describes the migration of DeepMoon from TensorFlow/Keras to PyTorch, completed in 2026.

## Summary of Changes

### 1. Dependencies (`requirements.txt`)
- **Removed**: TensorFlow 0.10.0rc0, Keras 1.2.2
- **Added**: PyTorch >= 1.9.0, torchvision >= 0.10.0
- All other dependencies remain unchanged

### 2. Model Architecture (`model_train.py`)

#### U-Net Implementation
The U-Net architecture has been reimplemented as a PyTorch `nn.Module`:
- **Parameters**: Exactly 10,278,017 parameters (matching original Keras model)
- **Architecture**: Identical encoder-decoder structure with skip connections
- **Activation**: ReLU activations in hidden layers, Sigmoid in output layer
- **Regularization**: L2 weight regularization and Dropout maintained

#### Key Classes
- `UNet`: PyTorch nn.Module implementing the U-Net architecture
- `CraterDataset`: PyTorch Dataset with data augmentation
- Data augmentation includes: horizontal/vertical flips, rotations, and pixel shifts

#### Training Loop
- Replaced Keras `fit_generator` with native PyTorch training loop
- Implemented manual early stopping based on validation loss
- Added proper L2 regularization to both training and validation loss
- Model checkpoints now include hyperparameters for reproducible inference

### 3. Model Serialization

#### File Format
- **Before**: `.h5` (HDF5 format used by Keras)
- **After**: `.pt` (PyTorch standard format)

#### Checkpoint Structure
```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch_number,
    'loss': validation_loss,
    'model_params': {
        'n_filters': 112,
        'FL': 3,
        'init': 'he_normal',
        'lmbda': 1e-6,
        'drop': 0.15
    }
}
```

### 4. Inference (`get_unique_craters.py`)
- Updated model loading to use PyTorch
- Reads model hyperparameters from checkpoint when available
- Falls back to default hyperparameters if not present in checkpoint
- Data preprocessing pipeline unchanged

### 5. Data Pipeline
- Input data generation (`input_data_gen.py`) unchanged
- Data preprocessing (`utils/processing.py`) unchanged
- All coordinate transformations and utilities remain compatible

## API Compatibility

### Training
The training script `run_model_train.py` maintains the same interface:
```python
MP = {
    'dir': 'catalogues/',
    'dim': 256,
    'bs': 8,
    'epochs': 4,
    'n_train': 30000,
    'n_dev': 5000,
    'n_test': 5000,
    'save_models': 1,
    'save_dir': 'models/model.pt',  # Changed from .h5 to .pt
    'N_runs': 1,
    'filter_length': [3],
    'lr': [0.0001],
    'n_filters': [112],
    'init': ['he_normal'],
    'lambda': [1e-6],
    'dropout': [0.15]
}
```

### Model Loading
```python
import torch
from model_train import UNet

# Load checkpoint
checkpoint = torch.load('models/model.pt', map_location='cpu')

# Extract hyperparameters
model_params = checkpoint.get('model_params', {
    'n_filters': 112,
    'FL': 3,
    'init': 'he_normal',
    'lmbda': 1e-6,
    'drop': 0.15
})

# Initialize and load model
model = UNet(**model_params)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Testing

### Unit Tests
- `tests/test_model_train.py` updated and passing
- Parameter count verified: 10,278,017 (matches original)
- All existing tests for data generation and utilities pass

### Integration Tests
All core functionalities tested and verified:
- ✅ Model building
- ✅ Forward pass with correct output shapes
- ✅ Data augmentation
- ✅ Training step with gradient computation
- ✅ Validation loss calculation
- ✅ Model saving/loading
- ✅ Prediction generation

## Performance Considerations

### GPU Support
- Automatic GPU detection and usage when available
- Falls back to CPU if GPU unavailable
- Model can be moved between devices as needed

### Memory Usage
- Similar memory footprint to original TensorFlow implementation
- 16GB GPU recommended for default batch size of 8
- Smaller batch sizes can be used for systems with less memory

## Migration Benefits

1. **Modern Framework**: PyTorch is actively maintained with strong community support
2. **Better Debugging**: Native Python execution allows easier debugging
3. **Flexibility**: More control over training loop and model behavior
4. **Reproducibility**: Hyperparameters saved with model checkpoints
5. **Consistency**: Validation loss now correctly includes L2 regularization

## Backward Compatibility

### Pre-trained Models
- Existing Keras models (.h5) cannot be directly loaded
- Models must be retrained with the PyTorch implementation
- Architecture and hyperparameters are identical, so results should be reproducible

### Data Files
- All training/validation/test data files remain compatible
- HDF5 data format unchanged
- Crater catalogues unchanged

## Known Limitations

1. Existing pre-trained Keras models must be retrained
2. The Jupyter notebook examples assume availability of PyTorch models
3. Python 2.7 is no longer supported (Python 3.5+ required)

## Future Enhancements

Potential improvements that could be made:
- Mixed precision training for faster training on modern GPUs
- Distributed training support for multi-GPU systems
- TensorBoard integration for training visualization
- Model quantization for faster inference
- ONNX export for deployment flexibility

## References

- Original Paper: [Lunar Crater Identification via Deep Learning](https://arxiv.org/abs/1803.02192)
- PyTorch Documentation: https://pytorch.org/docs/
- Original Repository: https://github.com/sarahchang2013/DeepMoon

## Authors

**Original Implementation (TensorFlow/Keras):**
- Ari Silburt
- Charles Zhu

**PyTorch Migration:**
- Completed: January 2026
- Migration maintained identical architecture and functionality
