# Roll Number: 727823TUAM033
# Evaluation Script for Product Review Sentiment Analysis

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import logging
from datetime import datetime
import joblib
from scipy import sparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print roll number and timestamp
print(f"Roll Number: 727823TUAM033")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

class ModelEvaluator:
    """
    Model evaluation class for comprehensive performance analysis
    """
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.tfidf_vectorizer = None
        
    def load_model_and_artifacts(self, model_path="../models/best_model.pkl", 
                                data_dir="../data/processed"):
        """
        Load trained model and necessary artifacts
        
        Args:
            model_path (str): Path to the trained model
            data_dir (str): Directory containing data artifacts
        """
        logger.info("Loading model and artifacts...")
        
        # Load the trained model
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load label encoder
        with open(os.path.join(data_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load TF-IDF vectorizer
        with open(os.path.join(data_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        logger.info("All artifacts loaded successfully")
    
    def load_test_data(self, data_dir="../data/processed"):
        """
        Load test data for evaluation
        
        Args:
            data_dir (str): Directory containing test data
            
        Returns:
            tuple: (X_test, y_test)
        """
        logger.info("Loading test data...")
        
        # Load test features and labels
        X_test = sparse.load_npz(os.path.join(data_dir, 'X_test.npz'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        logger.info(f"Test data loaded: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        return X_test, y_test
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            dict: Dictionary of all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted')
        }
        
        # Add AUC if probabilities are available
        if y_pred_proba is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                metrics['auc_roc'] = None
        
        return metrics
    
    def generate_classification_report(self, y_true, y_pred):
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            str: Classification report
        """
        # Get class names
        class_names = self.label_encoder.classes_
        
        # Generate report
        report = classification_report(y_true, y_pred, target_names=class_names)
        
        return report
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot and save confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path (str): Path to save the plot
        """
        # Get class names
        class_names = self.label_encoder.classes_
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path (str): Path to save the plot
        """
        if y_pred_proba is None:
            logger.warning("Probabilities not available, skipping ROC curve")
            return
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
        
        return fpr, tpr, roc_auc
    
    def evaluate_model_performance(self, X_test, y_test, save_plots=True):
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_plots (bool): Whether to save evaluation plots
            
        Returns:
            dict: Complete evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Get prediction probabilities if available
        y_pred_proba = None
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate classification report
        class_report = self.generate_classification_report(y_test, y_pred)
        
        # Create plots directory
        if save_plots:
            plots_dir = "../results/plots"
            os.makedirs(plots_dir, exist_ok=True)
        
        # Plot confusion matrix
        cm = self.plot_confusion_matrix(
            y_test, y_pred, 
            save_path=os.path.join(plots_dir, 'confusion_matrix.png') if save_plots else None
        )
        
        # Plot ROC curve
        roc_results = None
        if y_pred_proba is not None:
            roc_results = self.plot_roc_curve(
                y_test, y_pred_proba,
                save_path=os.path.join(plots_dir, 'roc_curve.png') if save_plots else None
            )
        
        # Compile results
        results = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'roc_results': roc_results,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        logger.info("Model evaluation completed successfully!")
        
        return results
    
    def save_evaluation_results(self, results, output_dir="../results"):
        """
        Save evaluation results to files
        
        Args:
            results (dict): Evaluation results
            output_dir (str): Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df.to_csv(os.path.join(output_dir, 'evaluation_metrics.csv'), index=False)
        
        # Save classification report
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(results['classification_report'])
        
        # Save confusion matrix
        cm_df = pd.DataFrame(results['confusion_matrix'], 
                            columns=self.label_encoder.classes_,
                            index=self.label_encoder.classes_)
        cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix.csv'))
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'true_labels': results['predictions'],  # Will be updated with actual test labels
            'predicted_labels': results['predictions']
        })
        predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
        
        logger.info(f"Evaluation results saved to {output_dir}")
    
    def print_evaluation_summary(self, results):
        """
        Print comprehensive evaluation summary
        
        Args:
            results (dict): Evaluation results
        """
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY")
        print("="*80)
        
        metrics = results['metrics']
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"  F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"  Precision (Weighted): {metrics['precision_weighted']:.4f}")
        print(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
        print(f"  Recall (Weighted): {metrics['recall_weighted']:.4f}")
        
        if metrics.get('auc_roc') is not None:
            print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        
        print(f"\nCLASSIFICATION REPORT:")
        print(results['classification_report'])
        
        print(f"\nCONFUSION MATRIX:")
        print(results['confusion_matrix'])
        
        print("="*80)
    
    def test_with_sample_reviews(self, sample_reviews):
        """
        Test the model with sample reviews
        
        Args:
            sample_reviews (list): List of sample review texts
        """
        logger.info("Testing model with sample reviews...")
        
        print("\n" + "="*80)
        print("SAMPLE REVIEW PREDICTIONS")
        print("="*80)
        
        for i, review in enumerate(sample_reviews, 1):
            # Preprocess the review (same as training)
            import re
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            from nltk.tokenize import word_tokenize
            
            # Download NLTK resources if needed
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            
            # Clean text
            text = review.lower()
            text = re.sub(r'[^a-z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenize and remove stopwords
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(word) for word in tokens 
                     if word not in stop_words and len(word) > 2]
            cleaned_text = ' '.join(tokens)
            
            # Transform with TF-IDF
            text_features = self.tfidf_vectorizer.transform([cleaned_text])
            
            # Predict
            prediction = self.model.predict(text_features)[0]
            prediction_proba = None
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(text_features)[0]
            
            # Convert back to original label
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            
            print(f"\nSample {i}:")
            print(f"  Review: {review}")
            print(f"  Predicted Sentiment: {predicted_label}")
            if prediction_proba is not None:
                print(f"  Confidence: {max(prediction_proba):.4f}")
        
        print("="*80)

def main():
    """
    Main function to execute evaluation pipeline
    """
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Load model and artifacts
        evaluator.load_model_and_artifacts()
        
        # Load test data
        X_test, y_test = evaluator.load_test_data()
        
        # Evaluate model
        results = evaluator.evaluate_model_performance(X_test, y_test, save_plots=True)
        
        # Save results
        evaluator.save_evaluation_results(results)
        
        # Print summary
        evaluator.print_evaluation_summary(results)
        
        # Test with sample reviews
        sample_reviews = [
            "This product is amazing! I love it so much.",
            "Terrible quality, waste of money.",
            "Good value for the price, would recommend.",
            "Not what I expected, very disappointed."
        ]
        evaluator.test_with_sample_reviews(sample_reviews)
        
        logger.info("Evaluation pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
