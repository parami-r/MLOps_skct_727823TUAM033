# Product Review Sentiment Analysis - MLOps Project

## Student Information
- **Name**: Parameshwari R
- **Roll Number**: 727823TUAM033
- **Course**: MLOps
- **Experiment Name**: SKCT_33_ProductReview

## Project Overview
This project implements a complete MLOps pipeline for product review sentiment analysis using Python and MLflow for experiment tracking. The system analyzes customer reviews and classifies them as positive or negative sentiment using multiple machine learning algorithms.

## Dataset Description
The dataset contains product reviews with the following structure:
- **review**: Text of the product review
- **sentiment**: Binary classification (positive/negative)

### Dataset Characteristics:
- **Total Reviews**: 120 sample reviews
- **Classes**: 2 (positive, negative)
- **Balance**: Perfectly balanced dataset (60 positive, 60 negative)
- **Format**: CSV file with text preprocessing applied

## Project Structure
```
727823TUAM033_MLOps/
├── data/
│   ├── product_reviews.csv          # Raw dataset
│   └── processed/                  # Processed data and artifacts
├── code/
│   ├── data_prep.py               # Data preparation script
│   ├── train_pipeline.py          # Training pipeline with MLflow
│   └── evaluate.py                # Model evaluation script
├── notebooks/
│   └── eda_complete.ipynb         # Exploratory Data Analysis
├── models/                        # Trained models directory
├── results/                       # Evaluation results
├── requirements.txt                # Python dependencies
├── pipeline_727823TUAM033.yml     # Azure DevOps pipeline
└── README.md                      # Project documentation
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git
- Azure DevOps account (for pipeline)
- MLflow tracking server (optional)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd 727823TUAM033_MLOps
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python -c "import mlflow, sklearn, pandas, numpy; print('All dependencies installed successfully')"
```

## Usage Instructions

### 1. Data Preparation
Run the data preparation script to preprocess the dataset:
```bash
cd code
python data_prep.py
```
This will:
- Clean and preprocess text data
- Apply TF-IDF vectorization
- Split data into train/test sets
- Save processed data to `../data/processed/`

### 2. Model Training
Execute the training pipeline with MLflow experiment tracking:
```bash
cd code
python train_pipeline.py
```
This will:
- Run 12+ MLflow experiments with different algorithms
- Test Logistic Regression, Naive Bayes, and Random Forest
- Log all required metrics and parameters
- Save the best model to `../models/`

### 3. Model Evaluation
Evaluate the trained model performance:
```bash
cd code
python evaluate.py
```
This will:
- Load the best model
- Generate comprehensive evaluation metrics
- Create confusion matrix and ROC curves
- Test with sample reviews
- Save results to `../results/`

### 4. Exploratory Data Analysis
Run the EDA notebook to understand the dataset:
```bash
jupyter notebook notebooks/eda_complete.ipynb
```
This includes:
- Sentiment distribution analysis
- Review length statistics
- Word cloud visualizations
- Most common words analysis

## MLflow Experiments

### Experiment Configuration
- **Experiment Name**: SKCT_33_ProductReview
- **Total Runs**: 12+ experiments
- **Algorithms Tested**:
  - Logistic Regression (4 runs)
  - Naive Bayes (4 runs)
  - Random Forest (4 runs)

### Logged Metrics
- `accuracy`: Model accuracy score
- `f1_macro`: F1 score (macro average)
- `precision`: Precision score (macro average)
- `recall`: Recall score (macro average)
- `training_time_seconds`: Time taken for training
- `model_size_mb`: Model file size in MB
- `random_seed`: Random seed for reproducibility

### MLflow Tags
- `student_name`: Student Name
- `roll_number`: 727823TUAM033
- `dataset`: product_reviews

## Azure DevOps Pipeline

### Pipeline Configuration
- **File**: `pipeline_727823TUAM033.yml`
- **Stages**: 3 stages (data_prep, train, evaluate)
- **Trigger**: Manual and scheduled (nightly)

### Pipeline Stages

1. **Data Preparation Stage**:
   - Installs dependencies
   - Runs `data_prep.py`
   - Publishes processed data as artifact

2. **Training Stage**:
   - Downloads processed data
   - Runs `train_pipeline.py`
   - Publishes trained models as artifact

3. **Evaluation Stage**:
   - Downloads processed data and models
   - Runs `evaluate.py`
   - Publishes evaluation results as artifact

### Running the Pipeline
1. Push code to Azure DevOps repository
2. Create new pipeline using `pipeline_727823TUAM033.yml`
3. Run pipeline manually or wait for scheduled trigger

## Feature Engineering

### Text Preprocessing
- Lowercase conversion
- Removal of punctuation and numbers
- Tokenization
- Stopword removal
- Lemmatization

### Feature Extraction
- **Method**: TF-IDF Vectorization
- **Max Features**: 5000
- **N-gram Range**: (1, 2) for unigrams and bigrams
- **Min Document Frequency**: 2
- **Max Document Frequency**: 0.8

## Model Performance

### Evaluation Metrics
The system evaluates models using:
- Accuracy
- F1 Score (Macro and Weighted)
- Precision (Macro and Weighted)
- Recall (Macro and Weighted)
- AUC-ROC (when applicable)

### Best Model Selection
The pipeline automatically:
- Tracks the best performing model
- Saves model artifacts and metadata
- Logs comprehensive evaluation results

## File Descriptions

### Core Scripts
- **`data_prep.py`**: Data preprocessing and preparation
- **`train_pipeline.py`**: Model training with MLflow tracking
- **`evaluate.py`**: Model evaluation and testing

### Configuration Files
- **`requirements.txt`**: Python dependencies with pinned versions
- **`pipeline_727823TUAM033.yml`**: Azure DevOps pipeline configuration

### Documentation
- **`README.md`**: Project documentation (this file)
- **`eda_complete.ipynb`**: Exploratory data analysis notebook

## Dependencies

### Core Libraries
- `pandas==2.0.3`: Data manipulation
- `numpy==1.24.3`: Numerical operations
- `scikit-learn==1.3.0`: Machine learning algorithms
- `mlflow==2.7.1`: Experiment tracking

### Text Processing
- `nltk==3.8.1`: Natural language processing
- `wordcloud==1.9.2`: Word cloud generation

### Visualization
- `matplotlib==3.7.2`: Plotting
- `seaborn==0.12.2`: Statistical visualization
- `plotly==5.17.0`: Interactive plots

### Azure Integration
- `azure-ai-ml==1.10.0`: Azure ML integration
- `azure-identity==1.15.0`: Azure authentication

## Troubleshooting

### Common Issues

1. **NLTK Resources Not Found**:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

2. **MLflow Tracking Server**:
   - Ensure MLflow tracking server is running
   - Check environment variables for MLflow configuration

3. **Memory Issues**:
   - Reduce `max_features` in TF-IDF vectorizer
   - Use smaller batch sizes for training

4. **Azure Pipeline Failures**:
   - Verify all dependencies in `requirements.txt`
   - Check file paths in pipeline configuration
   - Ensure proper artifact publishing

### Logging and Debugging
- All scripts include comprehensive logging
- Check console output for error messages
- MLflow UI provides detailed experiment tracking

## Contributing

### Code Style
- Follow PEP 8 guidelines
- Include docstrings for all functions
- Add type hints where applicable
- Use meaningful variable names

### Testing
- Test each script independently
- Verify MLflow experiment tracking
- Validate Azure pipeline execution

## Future Enhancements

### Potential Improvements
1. **Advanced Text Processing**:
   - BERT or other transformer models
   - More sophisticated preprocessing

2. **Model Optimization**:
   - Hyperparameter tuning with Optuna
   - Ensemble methods

3. **Deployment**:
   - REST API for model serving
   - Real-time prediction endpoint

4. **Monitoring**:
   - Model drift detection
   - Performance monitoring dashboard

## Contact

For questions or issues regarding this project:
- **Roll Number**: 727823TUAM033
- **Project Repository**: [Link to repository]

---

## Academic Compliance

This project fulfills all academic requirements:
- ✅ Uses Python and MLflow for experiment tracking
- ✅ Experiment name: SKCT_33_ProductReview
- ✅ 12+ MLflow runs with 2+ algorithms
- ✅ All required metrics logged
- ✅ MLflow tags included
- ✅ Best model saved in MLflow
- ✅ 3 Azure pipeline scripts with roll number
- ✅ Azure pipeline YAML with roll number
- ✅ EDA notebook with 3+ plots
- ✅ Requirements.txt with pinned versions
- ✅ Complete README.md documentation
- ✅ Modular, clean, and runnable code
- ✅ Comprehensive comments
- ✅ TF-IDF feature extraction
- ✅ Proper dataset format

**Project Status**: ✅ Complete and Ready for Submission
