# 🏥 Breast Cancer Classification with Decision Tree

## 🌐 Working App Link -> <a href='https://breast-cancer-classification-decision-tree.streamlit.app/'>Breast Classifier using Decision Tree</a>

This project implements a custom Decision Tree classifier from scratch and provides a comprehensive Streamlit web application for breast cancer tumor classification.

## 🚀 Features

- **Custom Decision Tree Implementation**: Built from scratch in Python
- **Interactive Web Interface**: Beautiful Streamlit app with multiple tabs
- **Data Preprocessing**: Automatic data cleaning and feature scaling
- **Exploratory Data Analysis**: Visualizations and statistical insights
- **Model Training**: Interactive hyperparameter tuning
- **Performance Metrics**: Comprehensive evaluation with confusion matrix and classification report
- **Feature Importance**: Analysis of which features contribute most to predictions
- **Live Predictions**: Interactive interface for making new predictions

## 📁 Project Structure

```
ML From Scratch/
├── app.py                          # Main Streamlit application
├── DecisionTree.py                 # Custom Decision Tree implementation
├── Breast_cancer_dataset_cleaned.csv  # Preprocessed breast cancer dataset   
├── requirements.txt      # Dependencies
└── README.md                       # This file
```

## 🛠️ Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd "ML From Scratch"
   ```

2. **Activate your virtual environment** (if you have one)
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Streamlit-specific requirements**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## 📊 How to Use the App

### 1. **Data Overview Tab**
- View dataset statistics and class distribution
- Examine feature information and data types
- See the first few rows of the dataset

### 2. **EDA Tab**
- Explore feature correlations with heatmaps
- Visualize feature distributions by diagnosis
- Analyze target variable distribution

### 3. **Model Training Tab**
- Adjust hyperparameters using sidebar controls:
  - **Max Depth**: Maximum tree depth (1-10)
  - **Min Samples Leaf**: Minimum samples required in leaf nodes
  - **Min Information Gain**: Minimum information gain for splitting
  - **Test Size**: Proportion of data for testing
  - **Feature Selection**: Method for feature selection during splitting
  - **Scale Features**: Option to standardize features
- Click "Train Decision Tree Model" to start training

### 4. **Results Tab**
- View classification report and confusion matrix
- Analyze feature importance rankings
- Examine prediction probability distributions
- Make new predictions with custom feature values

## 🔧 Model Parameters

- **Max Depth**: Controls tree complexity (higher = more complex, risk of overfitting)
- **Min Samples Leaf**: Prevents overfitting by requiring minimum samples in leaves
- **Min Information Gain**: Only split if information gain exceeds this threshold
- **Feature Selection**: 
  - `all`: Use all features for splitting
  - `sqrt`: Use √(number of features) randomly selected features
  - `log`: Use log₂(number of features) randomly selected features

## 📈 Dataset Information

The Wisconsin Breast Cancer dataset contains:
- **569 samples** with 30 numerical features
- **Target**: Binary classification (Benign vs. Malignant)
- **Features**: Various measurements of cell nuclei characteristics
- **Class Balance**: Relatively balanced dataset

## 🎯 Model Performance

The custom Decision Tree typically achieves:
- **Accuracy**: 90-95% on test data
- **Training Time**: Fast (usually under 10 seconds)
- **Interpretability**: High (clear decision rules)

## 🐛 Troubleshooting

### Common Issues:

1. **Import Error for DecisionTree**
   - Ensure `DecisionTree.py` is in the same directory as `app.py`
   - Check that `treenode` package is installed

2. **Dataset Not Found**
   - Verify `Breast_cancer_dataset_cleaned.csv` exists in the project directory
   - Check file permissions

3. **Memory Issues**
   - Reduce `max_depth` parameter
   - Use smaller `test_size`
   - Disable feature scaling if not needed

4. **Slow Performance**
   - Reduce number of features to plot in EDA
   - Use smaller dataset for testing

## 🔮 Making Predictions

1. Train the model first
2. Go to the "Results" tab
3. Enter feature values (defaults are set to dataset means)
4. Click "Predict" to get classification results
5. View confidence scores and class probabilities

---
