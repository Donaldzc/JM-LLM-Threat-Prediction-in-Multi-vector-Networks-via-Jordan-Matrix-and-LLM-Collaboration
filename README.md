Jordan Matrix-Based Threat Detection System (Weixiejiance)
📖 Overview
Weixiejiance is an advanced Network Intrusion Detection System (NIDS) designed to identify malicious network traffic with high precision. The core of this project is a novel deep learning architecture that integrates Jordan Matrix Decomposition and Prior Constraint Attention Mechanisms to model the cascading propagation of network threats.

This repository contains:

Core Model Implementation: PyTorch implementations of the Full JM Model and its ablation variants.
Ablation Studies: Rigorous experiments to validate the contribution of each module (Jordan Decomposition, Prior Constraints, High-Order Blocks).
Visualization Tools: Scripts to generate training curves and performance metrics.
Dataset Support: Preprocessing pipelines for standard datasets like CIC-IDS-2017 and UNSW-NB15.
✨ Key Features
Jordan Matrix Integration: Utilizes Jordan decomposition to capture complex feature dependencies and threat propagation patterns.
Prior Constraint Attention: An attention mechanism guided by domain knowledge to filter noise and focus on critical threat indicators.
High-Order Block Modeling: Captures multi-stage attack behaviors through high-order Jordan blocks.
Comprehensive Ablation Analysis: Includes baseline, no-decomposition, no-prior, and no-high-order variants to prove model efficacy.
Robust Data Preprocessing: Handles infinite values, missing data, and extreme outliers specifically for network traffic data.
🛠️ Tech Stack
Language: Python 3.8+
Deep Learning Framework: PyTorch
Data Processing: Pandas, NumPy, Scikit-learn
Visualization: Matplotlib
🚀 Installation
Prerequisites
Python 3.8 or higher
CUDA-compatible GPU (recommended for faster training)
Steps
Clone the repository:

bash
git clone https://github.com/Donaldzc/JM-LLM-Threat-Prediction-in-Multi-vector-Networks-via-Jordan-Matrix-and-LLM-Collaboration.git
cd weixiejiance
Install dependencies:

bash
pip install torch pandas numpy scikit-learn matplotlib
Prepare Data:

Download the CIC-IDS-2017 or UNSW-NB15 dataset.
Merge/preprocess the data into a CSV format (e.g., merged.csv).
Update the DATA_PATH variable in the Python scripts to point to your local data file.
💻 Usage
1. Run Ablation Experiments (CIC-IDS-2017)
The script 2.py contains the optimized model definitions and training loop. It compares five models:

Baseline: Pure data-driven MLP.
Ablation-1: Removes Jordan Decomposition.
Ablation-2: Removes JM Prior Constraints.
Ablation-3: Removes High-Order Jordan Blocks.
Full-Model: The complete proposed architecture.
bash
cd CIC-IDS-2017/JM-LLM/xiaorong
python 2.py
Output: The script will print Accuracy, F1-Score, and MSE for each model variant.

2. Run Experiments on UNSW-NB15
To test the model on the UNSW-NB15 dataset:

bash
cd UNSW-NB/jordan_LLM/xiaorong
python xiaorong.py
(Note: Ensure xiaorong.py is configured with the correct data path for UNSW-NB15)

3. Visualize Results
After training, use the drawing script to visualize the performance curves. Ensure you have a CSV file (e.g., modified_results.csv) containing columns: epoch, n, loss, train, val, test, rpa.

bash
cd draw
python modified_r_draw_eng.py
This will generate PNG images for Loss, Training Accuracy, Validation Accuracy, Test Accuracy, and RPA curves.

📊 Model Architecture Details
The Full JM Model consists of:

Jordan Encoder: Combines input features with previous state feedback (input_dim + 1) and applies a linear transformation followed by Tanh activation.
High-Order Block: Processes the encoded features to capture complex, non-linear relationships typical in multi-stage attacks.
Prior Attention Mechanism: Applies a sigmoid-weighted attention mask to the high-order features, enforcing prior knowledge constraints.
Classification Head: A final linear layer with Sigmoid activation for binary classification (Benign vs. Malicious).
🧪 Experimental Results
Note: Replace the table below with your actual experimental results.

Model Variant	Accuracy	F1-Score	MSE
Baseline	0.9xxx	0.9xxx	0.0xxx
Ablation-1 (No Jordan)	0.9xxx	0.9xxx	0.0xxx
Ablation-2 (No Prior)	0.9xxx	0.9xxx	0.0xxx
Ablation-3 (No High-Order)	0.9xxx	0.9xxx	0.0xxx
Full JM Model	0.9xxx	0.9xxx	0.0xxx
⚠️ Important Notes on Optimization
As noted in 2.py, initial experiments may show insignificant differences between models due to:

Overfitting: Simple MLPs can achieve high accuracy on balanced subsets, masking the benefits of complex structures.
Preprocessing: Aggressive normalization might remove subtle threat propagation features.
Recommendations:

Use Gradient Clipping (torch.nn.utils.clip_grad_norm_) to stabilize training.
Reduce EPOCHS and use early stopping to prevent overfitting.
Ensure the dataset retains enough complexity to benefit from Jordan Matrix modeling.
📄 License
This project is licensed under the MIT License.

📞 Contact
For questions or collaborations, please contact:

Author: [Cheng]
Email: [d202280654@hust.edu.cn]
