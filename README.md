# MLinAPP-FP01-14
 Group 14 in the course of MLinAPP at the poliTo working on project FP01

## Group Members
- s317661 Tcaciuc Claudiu Constantin
- s313848 Gabriele Tomatis
- s287345 Federico Mustich

## Paper
[Machine Learning in Applications FP01 - 2024/14 Project Report](Paper_FP01_2024_14.pdf)

## How to run the code
Clone the repository and create a virtual environment, then install libraries from the `requirements.txt` file.

The Dataset folder must be placed in the root of the project. as `.\dataset\`

Each notebook is self-contained and can be run independently. The notebooks are organized as follows:
- data preprocessing
- model training
- model evaluation

for each model to select the frequency of the data to use, you can change the variables `freq` inside the 3rd cell of each notebook.

## Models
The models we trained, that can be found in the `src` folder, are:
- Bayesian MLP
- Random Forest
- Isolation Forest
- Autoencoder
- LSTM-AD
- LSTM-AE
- RNN-EBM
- DAGMM
- XGBoost (only implmented, but too heavy to run on our machines)

## Training Environment
- CPU: Ryzen 7 5800X 8-core 16-thread 3.8GHz
- GPU: NVIDIA RTX 3060 12GB
- RAM: 32GB DDR4 3200MHz
- OS: Windows 11
- Python 3.11.2

## Test Results
Test Results can be found in the `.\src\models_html\` folder in the form of HTML files separated by frequency and model.