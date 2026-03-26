# Roll Number: 727823TUAM033
# Data Preparation Script for Product Review Sentiment Analysis

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print roll number and timestamp
print(f"Roll Number: 727823TUAM033")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

class DataPreprocessor:
    """
    Data preprocessing class for product review sentiment analysis
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = None
        self.label_encoder = None
        
    def download_nltk_resources(self):
        """Download required NLTK resources"""
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            logger.info("NLTK resources downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading NLTK resources: {e}")
            
    def clean_text(self, text):
        """
        Clean and preprocess text data
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                 if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def load_data(self, file_path):
        """
        Load dataset from CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Dataset loaded successfully: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
            
    def preprocess_data(self, df):
        """
        Preprocess the dataset
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        logger.info("Starting data preprocessing...")
        
        # Check for missing values
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        
        # Drop rows with missing values
        df = df.dropna()
        
        # Clean text data
        logger.info("Cleaning text data...")
        df['cleaned_review'] = df['review'].apply(self.clean_text)
        
        # Remove empty reviews after cleaning
        df = df[df['cleaned_review'].str.len() > 0]
        
        # Encode sentiment labels
        self.label_encoder = LabelEncoder()
        df['sentiment_encoded'] = self.label_encoder.fit_transform(df['sentiment'])
        
        logger.info(f"Data preprocessing completed. Final shape: {df.shape}")
        logger.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
        
        return df
    
    def create_tfidf_features(self, texts, max_features=5000, ngram_range=(1, 2)):
        """
        Create TF-IDF features from text data
        
        Args:
            texts (pd.Series): Text data
            max_features (int): Maximum number of features
            ngram_range (tuple): Range of n-grams
            
        Returns:
            scipy.sparse.csr_matrix: TF-IDF feature matrix
        """
        logger.info("Creating TF-IDF features...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.8
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        logger.info(f"TF-IDF features created: {tfidf_matrix.shape}")
        
        return tfidf_matrix
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            test_size (float): Proportion of data for testing
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train and test sets...")
        
        # Create TF-IDF features
        X = self.create_tfidf_features(df['cleaned_review'])
        y = df['sentiment_encoded']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Testing set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessed_data(self, X_train, X_test, y_train, y_test, output_dir):
        """
        Save preprocessed data and fitted objects
        
        Args:
            X_train, X_test, y_train, y_test: Split data
            output_dir (str): Directory to save the data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sparse matrices
        from scipy import sparse
        sparse.save_npz(os.path.join(output_dir, 'X_train.npz'), X_train)
        sparse.save_npz(os.path.join(output_dir, 'X_test.npz'), X_test)
        
        # Save labels
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        # Save fitted objects
        with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
            
        with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        logger.info(f"Preprocessed data saved to {output_dir}")

def main():
    """
    Main function to execute data preprocessing pipeline
    """
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Download NLTK resources
        preprocessor.download_nltk_resources()
        
        # Load data
        data_path = '../data/product_reviews.csv'
        df = preprocessor.load_data(data_path)
        
        # Preprocess data
        df_processed = preprocessor.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(df_processed)
        
        # Save preprocessed data
        output_dir = '../data/processed'
        preprocessor.save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir)
        
        # Save processed dataframe for reference
        df_processed.to_csv(os.path.join(output_dir, 'processed_reviews.csv'), index=False)
        
        logger.info("Data preprocessing pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in data preprocessing pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
