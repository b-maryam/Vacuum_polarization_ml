# Vacuum Polarization Analysis with ML

This repository combines traditional QED calculations with machine learning to analyze vacuum polarization effects.

## Features
- Feynman diagram generation
- Symbolic amplitude calculation
- ML-based amplitude prediction
- Decay rate estimation

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Generate Feynman diagram: `python src/feynman_diagram.py`
3. Train ML model: `python src/ml_model.py`
4. Run analytical calculations: `python src/analytical_calculations.py`

## Methodology
1. **Feynman Diagrams**: Matplotlib-based visualization of vacuum polarization
2. **QED Calculations**: Symbolic computation using SymPy
3. **ML Integration**: Neural network for amplitude prediction
4. **Decay Rates**: Derived from calculated amplitudes

## Limitations
- Simplified QED model for educational purposes
- Synthetic training data for ML component
- Dimensional regularization not fully implemented
