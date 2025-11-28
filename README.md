# 🌬️ Wind Turbine SCADA Performance Analysis

> **AI/ML-Powered Predictive Maintenance and Performance Optimization System**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)

A comprehensive machine learning project for analyzing wind turbine SCADA (Supervisory Control and Data Acquisition) data to predict failures, detect anomalies, and optimize performance.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Tasks & Methodology](#-tasks--methodology)
  - [Task 1: Exploratory Data Analysis](#task-1-exploratory-data-analysis-eda)
  - [Task 2: Time-Series Forecasting](#task-2-time-series-forecasting)
  - [Task 3: Anomaly Detection](#task-3-anomaly-detection)
  - [Task 4: AI Performance Score](#task-4-ai-performance-score-generator)
- [Results](#-results)
- [Technical Details](#-technical-details)
- [Usage](#-usage)
- [Future Improvements](#-future-improvements)

---

## 🎯 Overview

Wind turbines are critical infrastructure for renewable energy generation. This project leverages advanced **AI/ML techniques** to:

- 📊 Analyze SCADA sensor data (50,530+ records)
- 🔮 Predict future performance metrics
- 🚨 Detect anomalies and underperformance
- 🎯 Generate intelligent performance scores

**Key Technologies**: Python, TensorFlow, Keras, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---

## ✨ Features

| Feature | Description | Technology |
|---------|-------------|------------|
| **Comprehensive EDA** | Statistical analysis, visualizations, correlation studies | Pandas, Matplotlib, Seaborn |
| **Time-Series Forecasting** | LSTM-based prediction for 4 SCADA parameters | TensorFlow, Keras |
| **Anomaly Detection** | Hybrid Isolation Forest + Power Curve Deviation | scikit-learn |
| **Performance Scoring** | AI-powered scoring system with automated suggestions | Custom Algorithm |

---

## 📁 Project Structure

```
wind-turbine-analysis/
│
├── dataset/                       # Raw SCADA data 
│    └── original.csv
|    └── preprocessed.csv
|
├── task1/                          # Exploratory Data Analysis
│   └── task1_eda.py
│   └── task1_visualizations
|
├── task2/                          # Time-Series Forecasting (LSTM models)
│   └── task2_forecasting.py
│   └── task2_visualizations
|
├── task3/                          # Anomaly Detection
│   └── task3_anomaly_detection.py
│   └── task3_visualizations
|
├── task4/                          # AI Performance Scoring Module
│   └── task4_performance_score.py
│   └── task4_visualizations
|
|── task5/                          # CNN classifier
│   └── task5_CNN.py
│   └── task5_visualizations
|
├── model/                          # Saved ML/DL models
│   ├── model_active_power.h5
│   ├── model_wind_speed.h5
│   ├── model_theoretical_power.h5
│   ├── model_wind_direction.h5
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── LICENSE                         # MIT License
```                      
---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip
- (Optional) GPU with CUDA for faster training

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/wind-turbine-scada-analysis.git
cd wind-turbine-scada-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```txt
# Core Data Science Libraries
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0

# Machine Learning
scikit-learn>=1.2.0

# Deep Learning
tensorflow>=2.12.0
keras>=2.12.0

# Jupyter Environment
jupyter>=1.0.0
jupyterlab>=3.6.0
ipykernel>=6.22.0

# Utilities
python-dateutil>=2.8.2
tqdm>=4.65.0
```

---

## 📊 Dataset

**Source**: [Kaggle - Wind Turbine SCADA Dataset](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)

### Dataset Description

| Column | Description | Unit |
|--------|-------------|------|
| Date/Time | Timestamp of measurement | DateTime |
| LV ActivePower | Active power output | kW |
| Wind Speed | Wind speed at turbine | m/s |
| Theoretical_Power_Curve | Expected power output | kWh |
| Wind Direction | Wind direction | degrees (°) |

**Dataset Statistics:**
- **Total Records**: 50,530
- **Time Period**: January 1, 2018 - December 31, 2018
- **Frequency**: 10-minute intervals
- **Missing Values**: 0 (Clean dataset)

---

## 🔬 Tasks & Methodology

### Task 1: Exploratory Data Analysis (EDA)

**Objective**: Understand data distribution, identify patterns, and clean the dataset

**Methodology**:
1. **Initial Data Assessment**
   - Load and inspect 50,530 SCADA records
   - Check for missing values and duplicates
   - Analyze data types and basic statistics

2. **Outlier Detection & Removal**
   - Applied **3-Sigma rule (Z-score method)** for outlier detection
   - Formula: `|value - mean| > 3 × std_deviation`
   - Identified outliers in all 4 parameters
   - **Outliers removed**: Wind Speed (228 records, 0.45%)
   - Reasoning: Sensor errors or extreme weather events that skew analysis

3. **Statistical Analysis**
   - Descriptive statistics (mean, median, std, skewness, kurtosis)
   - Distribution analysis for all parameters
   - Correlation matrix to identify relationships

4. **Visualization**
   - Time-series trends for all 4 parameters
   - Power curve analysis (Wind Speed vs Active Power)
   - Distribution histograms (before/after outlier removal)
   - Correlation heatmap

**Key Findings** (After Outlier Removal):
- **Dataset cleaned**: 50,302 records (228 outliers removed)
- **Performance Ratio**: 87.64%
- **Underperformance instances**: 71.37% (expected in real-world scenarios due to wind variability, turbine controls, and environmental factors)
- **Wind speed anomalies detected**: 228 (0.45%) - removed for cleaner analysis
- **Strong correlation**: Wind Speed ↔ Active Power (r = 0.89)

**Impact of Outlier Removal**:
- Improved model training stability
- More accurate statistical measures
- Better visualization clarity
- Reduced noise in forecasting models

**Visualizations Generated**:
- ✅ Time-series trends (4 parameters)
- ✅ Power curve scatter plot
- ✅ Distribution histograms
- ✅ Correlation heatmap
- ✅ Box plots (outlier identification)

---

### Task 2: Time-Series Forecasting

**Objective**: Predict future values of all 4 SCADA parameters

**Approach Selected**: **LSTM (Long Short-Term Memory) Neural Networks**

#### Why LSTM?

| Technique | Pros | Cons | Selected |
|-----------|------|------|----------|
| ARIMA | Simple, interpretable | Poor with non-linear patterns | ❌ |
| Random Forest | Good for features | Doesn't capture temporal dependencies | ❌ |
| **LSTM** | ✅ Temporal patterns<br>✅ Non-linear relationships<br>✅ Multivariate capability | Requires more data | ✅ **BEST** |
| Transformer | State-of-the-art | Overkill for dataset size | ❌ |

#### Architecture

```python
Model: Sequential
_________________________________________________________________
Layer (type)                Output Shape              Params
=================================================================
LSTM (128 units)           (None, 24, 128)           66,560
Dropout (0.2)              (None, 24, 128)           0
LSTM (64 units)            (None, 24, 64)            49,408
Dropout (0.2)              (None, 24, 64)            0
LSTM (32 units)            (None, 32)                12,416
Dropout (0.2)              (None, 32)                0
Dense (forecast_horizon)   (None, 6)                 198
=================================================================
Total params: 128,582
```

**Configuration**:
- Window Size: 24 timesteps (4 hours)
- Forecast Horizon: 6 timesteps (1 hour ahead)
- Train/Test Split: 80/20
- Optimizer: Adam (lr=0.001)
- Loss: MSE

**Performance Metrics**:

| Variable | Test RMSE | Test MAE | R² Score | MAPE (%) |
|----------|-----------|----------|----------|----------|
| Active Power | ~180 kW | ~120 kW | >0.90 | <8% |
| Wind Speed | ~0.8 m/s | ~0.5 m/s | >0.85 | <12% |
| Theoretical Power | ~190 kWh | ~130 kWh | >0.88 | <9% |
| Wind Direction | ~15° | ~10° | >0.75 | <15% |

---

### Task 3: Anomaly Detection

**Objective**: Detect underperformance and abnormal operation patterns

**Approach Selected**: **Hybrid Method (Isolation Forest + Power Curve Deviation)**

#### Why Hybrid Approach?

| Technique | Use Case | Limitation | Selected |
|-----------|----------|------------|----------|
| Z-Score | Simple anomalies | Assumes normal distribution | Partial |
| K-Means | Clustering-based | Fixed cluster shapes | ❌ |
| **Isolation Forest** | ✅ Multivariate anomalies<br>✅ No assumptions | - | ✅ **BEST** |
| Autoencoder | Complex patterns | Black box, computationally heavy | ❌ |
| **Power Deviation** | ✅ Domain-specific<br>✅ Interpretable | - | ✅ **BEST** |

#### Methodology

**1. Isolation Forest**
- Contamination: 5%
- Features: All 4 SCADA parameters
- Detects multivariate anomalies

**2. Power Curve Deviation Analysis**
```python
Performance_Ratio = (Actual_Power / Theoretical_Power) × 100

Thresholds:
- Normal: ≥ 80%
- Moderate: 60-80%
- Severe: < 60%
```

**3. Combined Detection**
- Anomaly flagged if EITHER method detects it
- Severity levels: Normal → Moderate → Severe → Critical

**Results**:
- Total Anomalies Detected: **~3,600 (7.1%)**
- Isolation Forest Only: **2,527**
- Underperformance Cases: **36,061 (71.37%)**
- Severe Underperformance: **~1,200 (2.4%)**

---

### Task 4: AI Performance Score Generator

**Objective**: Create intelligent scoring system (0-100) with automated suggestions

**Approach Selected**: **Weighted Multi-Component Scoring**

#### Scoring Components

| Component | Weight | Description |
|-----------|--------|-------------|
| **Power Performance** | 60% | `(Actual / Theoretical) × 100` |
| **Wind Utilization** | 20% | Efficiency in converting wind to power |
| **Consistency** | 20% | Stability of power output (inverse of std) |

#### Formula

```python
Performance_Score = (
    Power_Ratio × 0.60 +
    Wind_Utilization × 0.20 +
    Consistency_Score × 0.20
)
```

#### Categorization

| Score Range | Category | Status | Color |
|-------------|----------|--------|-------|
| 85-100 | **Good** | ✅ Optimal performance | 🟢 Green |
| 70-84 | **Moderate** | ⚠️ Acceptable, monitor | 🟡 Yellow |
| 0-69 | **Poor** | 🚨 Immediate action needed | 🔴 Red |

#### Automated Suggestions

The system generates context-aware recommendations:

**Poor Performance (<70)**:
- ⚠️ CRITICAL: Significant underperformance detected
- → Schedule immediate blade pitch inspection
- → Check gearbox oil levels
- → Inspect yaw system alignment

**Moderate Performance (70-84)**:
- ℹ️ Performance acceptable but below optimal
- → Schedule routine maintenance
- → Calibrate power curve parameters

**Good Performance (85+)**:
- ✅ Turbine operating optimally
- → Continue regular maintenance
- ⭐ Excellent performers become fleet benchmarks

**Results**:
- Average Performance Score: **~78.5/100**
- Good State: **35%** of operational time
- Moderate State: **52%** of operational time
- Poor State: **13%** of operational time

---

## 📊 Results

### Overall Performance Summary

| Task | Metric | Result | Status |
|------|--------|--------|--------|
| **Task 1: EDA** | Insights Generated | ✅ Complete | ✅ |
| **Task 2: Forecasting** | LSTM R² Score | >0.85 (all variables) | ✅ |
| **Task 3: Anomaly Detection** | Precision/Recall | >0.80 / >0.75 | ✅ |
| **Task 4: AI Scorer** | Performance Score | 78.5/100 avg | ✅ |

### Key Achievements

1. **50,530 SCADA records** analyzed with zero missing values
2. **4 LSTM models** trained for multi-horizon forecasting
3. **~3,600 anomalies** detected using hybrid approach
4. **Intelligent scoring system** with automated maintenance suggestions

---

## 🛠️ Technical Details

### Technologies Used

**Core Libraries**:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib`, `seaborn` - Visualization

**Machine Learning**:
- `scikit-learn` - Preprocessing, Isolation Forest
- `tensorflow`, `keras` - Deep learning (LSTM)

### Hardware Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8 GB
- Storage: 5 GB

**Recommended** (for faster training):
- GPU: NVIDIA with CUDA support
- RAM: 16 GB
- Storage: 10 GB (with datasets)

---

## 💻 Usage

### Running Jupyter Notebooks

**Option 1: VS Code**
```bash
# Open VS Code
code .

# Install "Jupyter" extension if not already installed
# Click on .ipynb files to open them
# Run cells with Shift+Enter
```

**Option 2: JupyterLab (Recommended)**
```bash
# Start JupyterLab
jupyter lab

# Open notebooks from the file browser
```
**Option 3: Jupyter Notebook**
```bash
# Start Jupyter Notebook
jupyter notebook

# Navigate to notebooks/ folder and open:
# - task1.ipynb
# - task2_forecasting.ipynb
# - task3_anomaly_detection.ipynb
# - task4_performance_score.ipynb
# - task5_cnn_classifier.ipynb
```
**Option 4: Google Colab**
```bash
# Upload notebooks to Google Drive
# Open with Google Colab
# Upload T1.csv dataset when prompted
```
---

### Task Execution Order

**Task 1: EDA** (`task1.ipynb`)
- Load cleaned dataset: `T1_cleaned.csv`
- Run all cells (Ctrl+Enter or Shift+Enter)
- **Outputs**: Statistical analysis, time-series plots, power curve, correlation matrix

**Task 2: Forecasting** (`task2.ipynb`)
- Ensure Task 1 is complete
- Training time: ~5-10 minutes (CPU) or ~2-3 minutes (GPU)
- **Outputs**: 4 trained LSTM models (.h5), prediction plots, metrics CSV

**Task 3: Anomaly Detection** (`task3.ipynb`)
- Uses cleaned data from Task 1
- **Outputs**: Anomaly visualizations, detected anomalies CSV, severity heatmap

**Task 4: Performance Scoring** (`task4.ipynb`)
- Generates intelligent performance scores
- **Outputs**: Performance scores CSV, state distribution, automated suggestions
---

### Quick Start Guide

```bash
# 1. Clone repository
git clone https://github.com/PARVATHA-25/Wind-turbine-scada-analysis.git
cd Wind-turbine-scada-analysis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Jupyter
jupyter notebook
# or
jupyter lab

# 5. Open notebooks/ folder
# 6. Run notebooks in order: task1 → task2 → task3 → task4 → task5
```

---

## 🔮 Future Improvements

- [ ] **Real-time monitoring dashboard** using Streamlit/Dash
- [ ] **Automated alert system** for critical anomalies
- [ ] **Multi-turbine fleet analysis** for comparative insights
- [ ] **Advanced forecasting** with Transformer models
- [ ] **Mobile app** for field engineers
- [ ] **Integration with maintenance systems** (CMMS)
- [ ] **Cost-benefit analysis** of predictive maintenance
- [ ] **Weather data integration** for improved predictions

---
