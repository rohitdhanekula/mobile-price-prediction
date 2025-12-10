# 📱 Mobile Price Prediction System

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1+-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)](#)

**A machine learning-powered mobile phone price prediction system that predicts real monetary values using modern 2024 specifications with an interactive Streamlit web interface.**

[View Live Demo](#features) • [Quick Start](#quick-start) • [Project Structure](#project-structure)

</div>

---

## 🎯 Overview

This project demonstrates a complete machine learning pipeline for predicting mobile phone prices in the Indian market (₹8,000 - ₹1,50,000). Unlike traditional classification systems, this project provides **real monetary value predictions** based on 20 different mobile phone specifications including battery capacity, RAM, storage, camera quality, and display resolution.

**Key Achievement**: Achieved **90.75% accuracy** with a Random Forest Classifier on unseen test data.

---

## ✨ Features

### 🔮 **Real Price Predictions**
- Predicts actual price ranges with monetary values (₹8,000 - ₹1,50,000)
- Supports 4 price categories: Budget, Mid-Range, Premium, and Flagship

### 📊 **Interactive Dashboard**
- Beautiful Streamlit web interface for real-time predictions
- Dynamic input sliders for all 20 mobile specifications
- Real-time probability distribution visualization
- Feature importance analysis with interactive charts

### 🧠 **Advanced Analytics**
- Feature impact analysis showing which specs affect price the most
- Price optimization suggestions
- Price comparison across all categories
- Cross-validation metrics (CV Score: 90.62% ± 0.97%)

### 📱 **Modern Specifications**
- Battery: 3000-6000 mAh
- RAM: 4-16 GB
- Storage: 64GB-1TB
- Primary Camera: 12-108 MP
- Screen Resolution: HD to 4K
- And 15 more specifications

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 90.75% |
| **Cross-Validation Score** | 90.62% ± 0.97% |
| **Algorithm** | Random Forest Classifier |
| **Number of Features** | 20 |
| **Prediction Classes** | 4 (Budget, Mid-Range, Premium, Flagship) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rohitdhanekula/mobile-price-prediction.git
   cd mobile-price-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python model.py
   ```

4. **Launch the web application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   - Navigate to `http://localhost:8501`
   - Start predicting mobile prices!

### For Windows Users
```bash
setup.bat
```

---

## 📁 Project Structure

```
mobile-price-prediction/
├── app.py                      # Streamlit web application
├── model.py                    # Model training script
├── price_mapper.py             # Price calculation & mapping
├── requirements.txt            # Python dependencies
├── setup.py                    # Python setup script
├── setup.bat                   # Windows setup script
├── LICENSE                     # MIT License
├── README.md                   # This file
├── datasets/
│   ├── train.csv              # Training dataset (legacy)
│   ├── moderntrain.csv        # Training dataset (2024 specs)
│   └── test.csv               # Test dataset
└── mobilepricemodel.pkl       # Trained model (generated)
```

---

## 💡 How It Works

### 1. **Data Input**
Users provide 20 mobile phone specifications through interactive sliders:
- Display metrics (resolution, screen size)
- Performance specs (RAM, processor clock speed, cores)
- Camera specs (front & primary)
- Battery & connectivity features
- Physical properties (weight, depth)

### 2. **Feature Scaling**
Input features are normalized using the fitted scaler from training data to ensure consistent predictions.

### 3. **Model Prediction**
The trained Random Forest Classifier predicts:
- Price category (0: Budget, 1: Mid-Range, 2: Premium, 3: Flagship)
- Prediction probability for each category

### 4. **Price Mapping**
Predicted category is converted to actual price ranges using:
- Base price ranges for each category
- Feature-based adjustments based on specifications
- Normalized feature weights

---

## 📊 Price Categories

| Category | Price Range | Average Price | Target User |
|----------|-------------|----------------|-------------|
| **Budget** | ₹8,000 - ₹15,000 | ₹12,000 | Students, Budget-conscious users |
| **Mid-Range** | ₹15,000 - ₹30,000 | ₹23,000 | Professionals, Regular users |
| **Premium** | ₹30,000 - ₹60,000 | ₹47,000 | Power users, Gaming |
| **Flagship** | ₹60,000 - ₹1,50,000 | ₹1,10,000 | Tech enthusiasts, Professionals |

---

## 🔝 Top 10 Feature Importance

The model identifies these as the most important features for price prediction:

1. **RAM** (25%) - Memory capacity has highest impact
2. **Internal Memory** (20%) - Storage affects price significantly
3. **Battery Power** (15%) - Battery capacity influences pricing
4. **Primary Camera** (18%) - Camera quality is crucial
5. **Front Camera** (8%)
6. **Screen Height** (5%)
7. **Screen Width** (5%)
8. **Clock Speed** (4%)
9. And more...

---

## 🛠️ Technologies Used

### ML & Data Science
- **scikit-learn** - Machine Learning library
- **pandas** - Data manipulation
- **NumPy** - Numerical computing

### Visualization
- **Plotly** - Interactive visualizations
- **Matplotlib** - Static plots
- **Seaborn** - Statistical graphics

### Web Framework
- **Streamlit** - Interactive web interface
- **Python 3.8+** - Programming language

---

## 📝 Usage Examples

### Example 1: Budget Phone Prediction
- Battery: 4000 mAh
- RAM: 4 GB
- Storage: 64 GB
- Camera: 12 MP
- **Predicted Price**: ₹8,000 - ₹15,000

### Example 2: Flagship Phone Prediction
- Battery: 5500 mAh
- RAM: 12 GB
- Storage: 512 GB
- Camera: 80 MP
- **Predicted Price**: ₹60,000 - ₹1,50,000

---

## 🐛 Troubleshooting

### Problem: "Model file not found"
**Solution**: Run `python model.py` first to train and save the model

### Problem: "ModuleNotFoundError"
**Solution**: Install all dependencies: `pip install -r requirements.txt`

### Problem: "Port already in use"
**Solution**: Run on a different port: `streamlit run app.py --server.port 8502`

### Problem: "Dataset file not found"
**Solution**: Ensure `datasets/` folder exists with required CSV files

---

## 📚 Learning Resources

This project demonstrates:
- ✓ Machine Learning classification
- ✓ Data preprocessing & feature scaling
- ✓ Model evaluation & cross-validation
- ✓ Interactive web applications with Streamlit
- ✓ Data visualization with Plotly
- ✓ Feature importance analysis
- ✓ Real-world project structure

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License allows free use, modification, and distribution of this software with minimal restrictions.

---

## 📞 Contact

- **GitHub**: [@rohitdhanekula](https://github.com/rohitdhanekula)
- **Project**: [Mobile Price Prediction](https://github.com/rohitdhanekula/mobile-price-prediction)

---

## 🙏 Acknowledgments

- Built during internship project work
- Inspired by real-world e-commerce price prediction challenges
- Thanks to the scikit-learn, Streamlit, and Plotly communities

---

<div align="center">

### ⭐ If this project helped you, please consider giving it a star!

Made with ❤️ by Rohit Dhanekula

</div>
