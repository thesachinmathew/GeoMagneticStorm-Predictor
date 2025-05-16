#🌌 GeoMagneticStorm Predictor
GeoMagneticStorm Predictor is an advanced machine learning-based system designed to accurately predict geomagnetic storms using real-time and historical space weather data. This project combines scientific research, feature engineering, and state-of-the-art AI models (including Random Forest, XGBoost, LSTM, and LSTM-CNN hybrids) to forecast geomagnetic storm intensity and potential risks. It also includes a comparative graphical GUI that displays prediction results, model accuracy, and helps users interpret results effectively.

🌠 Table of Contents
Overview

Motivation

Objectives

System Architecture

Data Collection

Data Preprocessing

Feature Engineering

Modeling

Random Forest

XGBoost

LSTM

LSTM-CNN Hybrid

GUI Features

Model Comparison

Tools & Technologies

Installation & Usage

Results

Challenges

Future Work

References

License

🧠 Overview
Geomagnetic storms, caused by solar wind disturbances and coronal mass ejections (CMEs), can significantly affect satellite operations, GPS signals, and power grids. Timely and accurate predictions of these storms are vital to mitigate the impact on technological infrastructure.

This project aims to address this by applying machine learning and deep learning techniques on historical solar and geomagnetic data to build a predictor that can alert users before major disruptions occur.

🚀 Motivation
The sun, although 150 million kilometers away, has profound effects on Earth’s electromagnetic environment. Solar flares and CMEs can cause geomagnetic storms that lead to:

Satellite damage

GPS inaccuracies

Radio signal blackouts

Transformer failure in power grids

Despite the critical impact, prediction of geomagnetic storms remains a challenge due to the complexity of solar-terrestrial interactions. This project leverages AI to improve forecasting accuracy and deliver intuitive insights through a graphical user interface.

🎯 Objectives
Collect and preprocess geomagnetic and solar wind data

Train multiple machine learning and deep learning models

Compare their predictive accuracy

Develop a Python-based GUI that visualizes model performance and predictions

Deploy the system for educational and research purposes

🏗️ System Architecture
The system follows a modular architecture comprising:

Data Module – Loads solar and geomagnetic data (Dst, Kp index, solar wind speed, IMF Bz, etc.)

Preprocessing Module – Cleans, normalizes, and formats the data

Model Training Module – Trains Random Forest, XGBoost, LSTM, and LSTM-CNN models

Prediction Module – Outputs storm intensity prediction (mild, moderate, severe)

GUI Module – Displays predictions, accuracy comparisons, and model outputs in real-time

📊 Data Collection
The dataset includes:

Kp Index: Measures geomagnetic storm strength (0–9)

Dst Index: Indicates storm-related changes in Earth's magnetic field

Solar Wind Speed (Vsw)

IMF Bz component

Density & Temperature of solar wind particles

Data was sourced from:

NASA OMNIWeb

Kyoto World Data Center

NOAA Space Weather Prediction Center (SWPC)

Historical records from 2000 to 2024 were used, with hourly resolution.

🧹 Data Preprocessing
Steps involved:

Missing value imputation (using forward-fill, backward-fill)

Normalization using MinMaxScaler and StandardScaler

Feature selection using correlation analysis

Time-lagged features for LSTM models

Class balancing (oversampling minor storm cases)

The target variable was storm classification based on Kp values:

Kp < 4: No Storm

4 ≤ Kp < 6: Mild

6 ≤ Kp < 8: Moderate

Kp ≥ 8: Severe

🧮 Feature Engineering
Key engineered features:

Moving average of Bz over 3/6/12 hours

Rate of change of solar wind speed

Lag features (e.g., past 3 hours of Bz)

Solar Wind Dynamic Pressure

IMF total strength (Bt)

These helped boost the learning of both machine learning and sequence models.

🧠 Modeling
1. Random Forest
Easy to train, interpretable

Performs well with non-sequential, tabular data

Used as a baseline model

Accuracy: ~83%

2. XGBoost
Gradient-boosted decision trees

Regularization helps avoid overfitting

Feature importance easily extracted

Accuracy: ~86%

3. LSTM (Long Short-Term Memory)
Handles sequential data (time series)

Learns long-term dependencies

Input shape: (samples, time_steps, features)

Accuracy: ~89%

4. LSTM-CNN Hybrid
LSTM layer for temporal context

CNN layer for feature extraction

Combines benefits of both models

Accuracy: ~92%

All models were trained in Google Colab and later integrated into a VS Code-based GUI.

💻 GUI Features
Developed using Tkinter, the GUI offers:

Model Selection: Choose between RF, XGBoost, LSTM, and LSTM-CNN

Real-time Prediction: Input recent space weather values

Graphical Comparison: Bar chart showing accuracy of all models

Prediction Output: Displays severity level (None, Mild, Moderate, Severe)

Dark Themed Interface: For better readability and visual appeal

All outputs are displayed with clear indicators and labels for scientific and non-scientific audiences.

📈 Model Comparison
Model	Accuracy	Pros	Cons
Random Forest	83%	Fast, simple, interpretable	Less temporal insight
XGBoost	86%	Regularized, robust	Still non-sequential
LSTM	89%	Captures time-dependency	Requires more training time
LSTM-CNN	92%	Best accuracy, robust learning	Complex to tune

Accuracy measured on test set after training on ~80% of the data.

🧰 Tools & Technologies
Languages: Python

Libraries:

Pandas, NumPy – Data processing

Matplotlib, Seaborn – Visualization

Scikit-learn, XGBoost – ML Models

TensorFlow, Keras – Deep Learning

Tkinter – GUI

Platform: Google Colab (training), VS Code (GUI app)

⚙️ Installation & Usage
Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/geomagneticstorm-predictor.git
cd geomagneticstorm-predictor
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the GUI

bash
Copy
Edit
python geomagnetic_gui.py
(Optional): Use Google Colab for training new models using train_models.ipynb.

✅ Results
LSTM-CNN model achieved ~92% test accuracy

GUI allows prediction with just a few user inputs

Prediction of severe storms matched historical records with >90% accuracy

Feature importance highlighted the strong role of Bz and Vsw

Sample output:

yaml
Copy
Edit
Predicted Storm Severity: MODERATE
Model Used: LSTM-CNN
Accuracy: 92.3%
🧗 Challenges
Handling missing or inconsistent data from different sources

Time-alignment of solar and geomagnetic data

Balancing the dataset (severe storms are rare)

Optimizing LSTM and CNN layers without overfitting

🌱 Future Work
Integrate real-time API from NOAA for live prediction

Add SVM and GRU models for extended comparison

Deploy as a Flask web app or Streamlit dashboard

Add alert system (email or SMS) for high-severity predictions

Apply transfer learning from solar cycle patterns

📚 References
NOAA SWPC (https://www.swpc.noaa.gov/)

OMNIWeb (https://omniweb.gsfc.nasa.gov/)

Kyoto WDC for Geomagnetism

NASA’s Space Weather Prediction

Related academic papers and IEEE journals

📝 License
This project is licensed under the MIT License — feel free to use, modify, and distribute it with credit.

GeoMagneticStorm Predictor combines space science and AI to create a reliable, user-friendly forecasting system. Whether you’re a researcher, data scientist, student, or space weather enthusiast, this tool helps bridge the gap between solar physics and real-world applications. 🚀🛰️
