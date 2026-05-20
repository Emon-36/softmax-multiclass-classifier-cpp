# Multiclass Softmax Classifier in C++

A 4-class softmax classifier built from scratch in pure C++ — no frameworks, no libraries, no dynamic memory allocation.

## What it does
- Classifies 4-feature input data into 4 distinct pattern classes
- Trained using Batch Gradient Descent with Categorical Cross-Entropy loss
- Interactive inference mode — enter any 4-element sample and get real-time prediction

## Architecture
- 4 input features → 4 output neurons
- Numerically stable Softmax (max-value subtraction)
- Batch gradient accumulation with averaged weight updates
- Epsilon-clamped Cross-Entropy to prevent log(0) NaN

## Results
| Metric | Value |
|--------|-------|
| Epochs | 900 |
| Initial Loss | 3.01 |
| Final Loss | 0.0015 |
| Test Accuracy | 99.8%+ |

## Classes
| Class | Pattern |
|-------|---------|
| 0 | HIGH, LOW, HIGH, LOW |
| 1 | HIGH, HIGH, LOW, LOW |
| 2 | LOW, HIGH, HIGH, LOW |
| 3 | LOW, LOW, HIGH, HIGH |

## Build & Run
```bash
g++ multiclass.cpp -o classifier
./classifier
```

## Why no frameworks?
Edge hardware — bare-metal ARM SoCs, microcontrollers — often can't run PyTorch or TensorFlow. This project demonstrates that a fully functional ML classifier can run anywhere a C++ compiler exists.

## Related
- [Binary Classifier in C++](https://github.com/Emon-36/edge-miso-classifier-cpp.git)
