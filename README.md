# Autoencoder-Based Anomaly Detection with Explainable Insights for Smart Grid Faults

## Problem Statement
Smart grids require fast, accurate, and explainable fault detection to maintain reliability and avoid system failures. Traditional fault detection methods may overlook minor warning signs that could lead to significant problems. Moreover, understanding the cause of faults is essential for improved maintenance and trust in AI-driven systems.

## Project Objective
This project aims to build an autoencoder-based anomaly detection system that identifies faults using simulated smart grid sensor data. Additionally, it incorporates an Explainable AI (XAI) technique to provide insights into why anomalies are detected, enhancing fault diagnosis and transparency.

---

## Technologies & Tools

### Autoencoders
- Learn normal behavior in sensor data.
- Detect anomalies based on reconstruction error.
- High reconstruction error signals a potential fault.

### Explainable AI (XAI)
- Helps interpret why a data point is classified as an anomaly.
- Improves human trust and enables better fault diagnosis.

### NumPy & Pandas
- Generate and manipulate synthetic sensor data.
- Handle data types like voltage, current, and frequency readings.

### Data Normalization & Scaling
- Standardize sensor readings for improved model performance.

### Matplotlib & Seaborn
- Visualize:
  - Sensor time series data.
  - Detected anomalies.
  - XAI outputs to explain fault predictions.

---

## Features
- Simulated smart grid environment.
- Autoencoder model for anomaly detection.
- Integrated explainability for model predictions.
- Interactive and visual analysis for fault interpretation.

---


## License
This project is open-source under the MIT License.
