
!pip install sympy --upgrade

import matplotlib.pyplot as plt
import os
import sympy as sp
# Import Idx from the correct location
# from sympy.tensor.index import Idx # This line caused the error
from sympy import tensor # import the tensor module
# You can then access Idx through the tensor module:
Idx = tensor.Idx

def create_feynman_diagram():
    fig, ax = plt.subplots()

    # External photon lines
    ax.arrow(0.2, 0.5, 0.3, 0, head_width=0.03, head_length=0.03, fc='blue')
    ax.arrow(0.8, 0.5, 0.3, 0, head_width=0.03, head_length=0.03, fc='blue')

    # Loop
    circle = plt.Circle((0.5, 0.5), 0.2, color='red', fill=False, linestyle='dashed')
    ax.add_artist(circle)

    plt.axis('off')

    # Create the 'diagrams' directory if it doesn't exist
    os.makedirs('diagrams', exist_ok=True)

    plt.savefig('diagrams/feynman_vacuum_polarization.png')
    plt.close()

if __name__ == "__main__":
    create_feynman_diagram()

def calculate_amplitude():
    # Define variables
    m, e, α = sp.symbols('m e alpha')
    # Use tensor.Idx instead of TensorIndex
    k = tensor.Idx('k')

    # Vacuum polarization tensor
    Π = (α/(3*sp.pi)) * (sp.log((k**2)/(4*m**2)) - 5/3)

    return Π

if __name__ == "__main__":
    print("Symbolic Amplitude Expression:")
    print(calculate_amplitude())

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def generate_training_data(num_samples=10000):
    # Generate synthetic training data (k^2 values in GeV^2)
    k_sq = np.linspace(0.1, 10, num_samples)
    amplitudes = 0.1 * np.log(k_sq/4) - 0.5  # Simplified model

    return k_sq.reshape(-1,1), amplitudes

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def train_model():
    X, y = generate_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_model()
    model.fit(X_train, y_train, epochs=50, validation_split=0.2)
    model.save('models/trained_amplitude_predictor.h5')
    return model

if __name__ == "__main__":
    train_model()

import numpy as np
from sympy import symbols, integrate

def calculate_decay_rate(amplitude):
    # Simplified decay rate calculation
    α, m = symbols('alpha m')
    Γ = (α**2 * m) / (48 * np.pi) * abs(amplitude)**2
    return Γ

if __name__ == "__main__":
    amp = 0.001  # Example amplitude value
    print(f"Decay Rate: {calculate_decay_rate(amp)}")
