import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from DecisionTree import DecisionTree
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Set page config
st.set_page_config(
    page_title="Breast Cancer Classification with Decision Tree",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Breast Cancer Classification with Decision Tree")
st.markdown("""
This app uses a custom Decision Tree implementation to classify breast cancer tumors as benign or malignant.
The model is trained on the Wisconsin Breast Cancer dataset and provides interactive analysis tools.
""")

# Load and preprocess data
@st.cache_data
def load_data():
    """Load and preprocess the breast cancer dataset"""
    try:
        # Load the cleaned dataset
        df = pd.read_csv('Breast_cancer_dataset_cleaned.csv')
        
        # Convert diagnosis to binary (M=1, B=0)
        df['diagnosis_binary'] = df['diagnosis'].map({'M': 1, 'B': 0})
        
        # Ensure proper data types for Streamlit compatibility
        # Keep diagnosis as string to avoid Arrow conversion issues
        df['diagnosis'] = df['diagnosis'].astype(str)
        
        # Ensure all numeric columns are float64 for better compatibility
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'diagnosis_binary':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
df = load_data()

if df is not None:
    st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    
    # Get numeric columns for the entire app
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'diagnosis_binary']
    
    # Sidebar for model parameters
    st.sidebar.header("🔧 Model Parameters")
    
    # Model hyperparameters
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 4)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 1)
    min_information_gain = st.sidebar.slider("Min Information Gain", 0.0, 0.5, 0.0, 0.01)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random State", 0, 100, 42)
    
    # Feature selection method
    feature_selection = st.sidebar.selectbox(
        "Feature Selection Method",
        ["all", "sqrt", "log"],
        help="Method for selecting features during splitting"
    )
    
    # Data preprocessing options
    st.sidebar.header("📊 Data Preprocessing")
    scale_features = st.sidebar.checkbox("Scale Features", value=True)
    show_eda = st.sidebar.checkbox("Show Exploratory Data Analysis", value=True)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Data Overview", "🔍 EDA", "🤖 Model Training", "📊 Results"])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Info")
            st.write(f"**Total samples:** {len(df)}")
            st.write(f"**Features:** {len(df.columns) - 2}")  # Excluding diagnosis and diagnosis_binary
            st.write(f"**Target variable:** diagnosis (M=Malignant, B=Benign)")
            
            # Class distribution
            diagnosis_counts = df['diagnosis'].value_counts()
            st.write("**Class Distribution:**")
            st.write(f"Benign (B): {diagnosis_counts.get('B', 0)} samples")
            st.write(f"Malignant (M): {diagnosis_counts.get('M', 0)} samples")
            
            # Calculate class balance
            total = len(df)
            benign_ratio = diagnosis_counts.get('B', 0) / total
            malignant_ratio = diagnosis_counts.get('M', 0) / total
            st.write(f"**Class Balance:**")
            st.write(f"Benign: {benign_ratio:.1%}")
            st.write(f"Malignant: {malignant_ratio:.1%}")
        
        with col2:
            st.subheader("Feature Statistics")
            # Ensure we have numeric columns
            if len(numeric_cols) > 0:
                # Show basic statistics for first few features
                st.write(df[numeric_cols[:5]].describe())
            else:
                st.warning("No numeric columns found. Please check your dataset.")
                st.write("Available columns:", list(df.columns))
                st.write("Column dtypes:", df.dtypes)
        
        # Show first few rows
        st.subheader("First 10 Rows")
        try:
            # Convert to string representation to avoid Arrow conversion issues
            display_df = df.head(10).copy()
            st.dataframe(display_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying dataframe: {e}")
            st.write("Raw data preview:")
            st.write(df.head(10).to_dict())
        
        # Show data types
        st.subheader("Data Types")
        st.write(df.dtypes)
        
        # Show data info
        st.subheader("Data Info")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
    
    with tab2:
        if show_eda:
            st.header("Exploratory Data Analysis")
            
            # Correlation heatmap
            st.subheader("Feature Correlation Heatmap")
            try:
                correlation_matrix = df[numeric_cols].corr()
                
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
                plt.title('Feature Correlation Heatmap')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Error creating correlation heatmap: {e}")
                st.write("Correlation matrix (raw):")
                st.write(correlation_matrix)
            
            # Feature distributions
            st.subheader("Feature Distributions by Diagnosis")
            
            # Select features to plot
            features_to_plot = st.multiselect(
                "Select features to visualize:",
                numeric_cols[:10],  # Limit to first 10 features
                default=numeric_cols[:5]
            )
            
            if features_to_plot:
                try:
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    axes = axes.flatten()
                    
                    for i, feature in enumerate(features_to_plot[:6]):
                        if i < len(axes):
                            df.boxplot(column=feature, by='diagnosis', ax=axes[i])
                            axes[i].set_title(f'{feature} by Diagnosis')
                            axes[i].set_xlabel('Diagnosis')
                            axes[i].set_ylabel(feature)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Error creating boxplots: {e}")
                    st.write("Feature distributions (raw data):")
                    for feature in features_to_plot[:6]:
                        st.write(f"**{feature}:**")
                        st.write(df.groupby('diagnosis')[feature].describe())
            
            # Target distribution pie chart
            st.subheader("Target Distribution")
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                # Convert to regular pandas Series for plotting
                diagnosis_counts_plot = pd.Series(diagnosis_counts.values, index=diagnosis_counts.index)
                diagnosis_counts_plot.plot(kind='pie', autopct='%1.1f%%', startangle=90)
                plt.title('Distribution of Diagnosis')
                plt.ylabel('')
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Error creating pie chart: {e}")
                st.write("Diagnosis distribution (raw):")
                st.write(diagnosis_counts)
    
    with tab3:
        st.header("Model Training")
        
        if st.button("🚀 Train Decision Tree Model"):
            with st.spinner("Training the model..."):
                try:
                    # Prepare features and target
                    X = df[numeric_cols].values
                    y = df['diagnosis_binary'].values
                    
                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                    
                    # Scale features if requested
                    if scale_features:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        X_train_final = X_train_scaled
                        X_test_final = X_test_scaled
                    else:
                        X_train_final = X_train
                        X_test_final = X_test
                    
                    # Initialize and train the model
                    try:
                        model = DecisionTree(
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            min_information_gain=min_information_gain,
                            numb_of_features_splitting=feature_selection
                        )
                        
                        model.train(X_train_final, y_train)
                    except Exception as e:
                        st.error(f"❌ Error initializing or training model: {e}")
                        st.error("Please check your DecisionTree implementation and parameters.")
                        st.stop()
                    
                    # Make predictions
                    y_pred = model.predict(X_test_final)
                    y_pred_proba = model.predict_proba(X_test_final)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Store results in session state
                    st.session_state.model = model
                    st.session_state.X_test = X_test_final
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = y_pred
                    st.session_state.y_pred_proba = y_pred_proba
                    st.session_state.accuracy = accuracy
                    st.session_state.feature_names = numeric_cols
                    st.session_state.scaler = scaler if scale_features else None
                    
                    st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.3f}")
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {e}")
                    st.error("Please check your parameters and try again.")
        
        # Show training info if model exists
        if 'model' in st.session_state:
            st.subheader("Model Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Max Depth:** {st.session_state.model.max_depth}")
                st.write(f"**Min Samples Leaf:** {st.session_state.model.min_samples_leaf}")
                st.write(f"**Min Information Gain:** {st.session_state.model.min_information_gain}")
                st.write(f"**Feature Selection:** {st.session_state.model.numb_of_features_splitting}")
            
            with col2:
                st.write(f"**Training Accuracy:** {st.session_state.accuracy:.3f}")
                st.write(f"**Test Set Size:** {len(st.session_state.X_test)}")
                st.write(f"**Number of Features:** {len(st.session_state.feature_names)}")
                st.write(f"**Features Scaled:** {'Yes' if st.session_state.scaler else 'No'}")
    
    with tab4:
        st.header("Model Results")
        
        if 'model' in st.session_state:
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(
                st.session_state.y_test, 
                st.session_state.y_pred, 
                target_names=['Benign', 'Malignant'],
                output_dict=True
            )
            
            # Display as a nice table
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Benign', 'Malignant'],
                       yticklabels=['Benign', 'Malignant'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            st.pyplot(fig)
            plt.close()
            
            # Feature importance
            st.subheader("Feature Importance")
            if hasattr(st.session_state.model, 'feature_importances'):
                feature_importance_df = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Importance': list(st.session_state.model.feature_importances.values())
                }).sort_values('Importance', ascending=False)
                
                # Plot top 15 features
                top_features = feature_importance_df.head(15)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.barplot(data=top_features, x='Importance', y='Feature')
                plt.title('Top 15 Most Important Features')
                plt.xlabel('Feature Importance')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Show feature importance table
                st.dataframe(feature_importance_df)
            
            # Prediction probabilities distribution
            st.subheader("Prediction Probabilities Distribution")
            if hasattr(st.session_state, 'y_pred_proba'):
                proba_df = pd.DataFrame({
                    'True_Label': ['Benign' if x == 0 else 'Malignant' for x in st.session_state.y_test],
                    'Predicted_Probability_Malignant': st.session_state.y_pred_proba[:, 1]
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                for label in ['Benign', 'Malignant']:
                    data = proba_df[proba_df['True_Label'] == label]['Predicted_Probability_Malignant']
                    plt.hist(data, alpha=0.7, label=label, bins=20)
                
                plt.xlabel('Predicted Probability of Malignant')
                plt.ylabel('Frequency')
                plt.title('Distribution of Prediction Probabilities')
                plt.legend()
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            # Interactive prediction
            st.subheader("🔮 Make New Predictions")
            st.write("Enter feature values to make a prediction:")
            
            # Create input form for new predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Mean Values:**")
                feature_inputs = {}
                # Handle cases where we have fewer than 15 features
                features_first_col = st.session_state.feature_names[:min(15, len(st.session_state.feature_names))]
                for i, feature in enumerate(features_first_col):
                    try:
                        default_val = float(df[feature].mean())
                        feature_inputs[feature] = st.number_input(
                            feature, 
                            value=default_val, 
                            format="%.3f",
                            key=f"input_{i}"
                        )
                    except Exception as e:
                        st.error(f"Error with feature {feature}: {e}")
                        feature_inputs[feature] = 0.0
            
            with col2:
                st.write("**Standard Error Values:**")
                # Handle cases where we have fewer than 30 features
                if len(st.session_state.feature_names) > 15:
                    features_second_col = st.session_state.feature_names[15:min(30, len(st.session_state.feature_names))]
                    for i, feature in enumerate(features_second_col):
                        try:
                            default_val = float(df[feature].mean())
                            feature_inputs[feature] = st.number_input(
                                feature, 
                                value=default_val, 
                                format="%.3f",
                                key=f"input_{i+15}"
                            )
                        except Exception as e:
                            st.error(f"Error with feature {feature}: {e}")
                            feature_inputs[feature] = 0.0
                else:
                    st.write("No additional features for second column")
            
            if st.button("🔮 Predict"):
                try:
                    # Prepare input data
                    input_data = np.array([[feature_inputs[feature] for feature in st.session_state.feature_names]])
                    
                    # Scale if needed
                    if st.session_state.scaler:
                        input_data = st.session_state.scaler.transform(input_data)
                    
                    # Make prediction
                    prediction = st.session_state.model.predict(input_data)[0]
                    probabilities = st.session_state.model.predict_proba(input_data)[0]
                    
                    # Display results
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.error("🚨 **Prediction: MALIGNANT**")
                        else:
                            st.success("✅ **Prediction: BENIGN**")
                        
                        st.write(f"**Confidence:** {max(probabilities):.1%}")
                    
                    with col2:
                        st.write("**Class Probabilities:**")
                        st.write(f"Benign: {probabilities[0]:.1%}")
                        st.write(f"Malignant: {probabilities[1]:.1%}")
                    
                    # Create probability bar chart
                    fig, ax = plt.subplots(figsize=(8, 6))
                    classes = ['Benign', 'Malignant']
                    colors = ['green', 'red']
                    bars = plt.bar(classes, probabilities, color=colors, alpha=0.7)
                    plt.title('Prediction Probabilities')
                    plt.ylabel('Probability')
                    plt.ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar, prob in zip(bars, probabilities):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                f'{prob:.1%}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {e}")
        
        else:
            st.info("👆 Please train a model first in the 'Model Training' tab!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit and Custom Decision Tree Implementation</p>
        <p>Dataset: Wisconsin Breast Cancer Dataset</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Failed to load dataset. Please check if 'Breast_cancer_dataset_cleaned.csv' exists in the current directory.") 