# Roll Number: 727823TUAM033
# Training Pipeline Script for Product Review Sentiment Analysis

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import pickle
import os
import time
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

class ModelTrainer:
    """
    Model training class with MLflow experiment tracking
    """
    
    def __init__(self, experiment_name="SKCT_33_ProductReview"):
        self.experiment_name = experiment_name
        self.best_model = None
        self.best_score = 0
        self.best_run_id = None
        
        # Set MLflow tracking
        mlflow.set_experiment(experiment_name)
        
        # Student information for MLflow tags
        self.student_info = {
            "student_name": "Student Name",
            "roll_number": "727823TUAM033",
            "dataset": "product_reviews"
        }
        
    def load_data(self, data_dir="../data/processed"):
        """
        Load preprocessed data
        
        Args:
            data_dir (str): Directory containing preprocessed data
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, label_encoder)
        """
        logger.info("Loading preprocessed data...")
        
        # Load sparse matrices
        X_train = sparse.load_npz(os.path.join(data_dir, 'X_train.npz'))
        X_test = sparse.load_npz(os.path.join(data_dir, 'X_test.npz'))
        
        # Load labels
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        # Load label encoder
        with open(os.path.join(data_dir, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test, label_encoder
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro')
        }
        
        return metrics
    
    def calculate_model_size(self, model):
        """
        Calculate model size in MB
        
        Args:
            model: Trained model
            
        Returns:
            float: Model size in MB
        """
        # Save model to temporary file to get size
        temp_file = 'temp_model.pkl'
        joblib.dump(model, temp_file)
        size_mb = os.path.getsize(temp_file) / (1024 * 1024)
        os.remove(temp_file)
        return size_mb
    
    def train_logistic_regression(self, X_train, X_test, y_train, y_test, params):
        """
        Train Logistic Regression model
        
        Args:
            X_train, X_test, y_train, y_test: Data splits
            params: Model parameters
            
        Returns:
            tuple: (model, metrics)
        """
        with mlflow.start_run(run_name=f"LogisticRegression_{params['C']}_{params['solver']}") as run:
            # Log parameters
            mlflow.log_params(params)
            mlflow.set_tags(self.student_info)
            
            # Start training timer
            start_time = time.time()
            
            # Train model
            model = LogisticRegression(
                C=params['C'],
                solver=params['solver'],
                max_iter=params['max_iter'],
                random_state=params['random_seed']
            )
            model.fit(X_train, y_train)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test)
            metrics['training_time_seconds'] = training_time
            metrics['model_size_mb'] = self.calculate_model_size(model)
            metrics['random_seed'] = params['random_seed']
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Track best model
            if metrics['accuracy'] > self.best_score:
                self.best_score = metrics['accuracy']
                self.best_model = model
                self.best_run_id = run.info.run_id
            
            logger.info(f"Logistic Regression - Accuracy: {metrics['accuracy']:.4f}, "
                       f"F1: {metrics['f1_macro']:.4f}, Time: {training_time:.2f}s")
            
            return model, metrics
    
    def train_naive_bayes(self, X_train, X_test, y_train, y_test, params):
        """
        Train Naive Bayes model
        
        Args:
            X_train, X_test, y_train, y_test: Data splits
            params: Model parameters
            
        Returns:
            tuple: (model, metrics)
        """
        with mlflow.start_run(run_name=f"NaiveBayes_{params['alpha']}") as run:
            # Log parameters
            mlflow.log_params(params)
            mlflow.set_tags(self.student_info)
            
            # Start training timer
            start_time = time.time()
            
            # Train model
            model = MultinomialNB(
                alpha=params['alpha'],
                fit_prior=params['fit_prior']
            )
            model.fit(X_train, y_train)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test)
            metrics['training_time_seconds'] = training_time
            metrics['model_size_mb'] = self.calculate_model_size(model)
            metrics['random_seed'] = params['random_seed']
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Track best model
            if metrics['accuracy'] > self.best_score:
                self.best_score = metrics['accuracy']
                self.best_model = model
                self.best_run_id = run.info.run_id
            
            logger.info(f"Naive Bayes - Accuracy: {metrics['accuracy']:.4f}, "
                       f"F1: {metrics['f1_macro']:.4f}, Time: {training_time:.2f}s")
            
            return model, metrics
    
    def train_random_forest(self, X_train, X_test, y_train, y_test, params):
        """
        Train Random Forest model
        
        Args:
            X_train, X_test, y_train, y_test: Data splits
            params: Model parameters
            
        Returns:
            tuple: (model, metrics)
        """
        with mlflow.start_run(run_name=f"RandomForest_{params['n_estimators']}_{params['max_depth']}") as run:
            # Log parameters
            mlflow.log_params(params)
            mlflow.set_tags(self.student_info)
            
            # Start training timer
            start_time = time.time()
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                random_state=params['random_seed']
            )
            model.fit(X_train, y_train)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test)
            metrics['training_time_seconds'] = training_time
            metrics['model_size_mb'] = self.calculate_model_size(model)
            metrics['random_seed'] = params['random_seed']
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Track best model
            if metrics['accuracy'] > self.best_score:
                self.best_score = metrics['accuracy']
                self.best_model = model
                self.best_run_id = run.info.run_id
            
            logger.info(f"Random Forest - Accuracy: {metrics['accuracy']:.4f}, "
                       f"F1: {metrics['f1_macro']:.4f}, Time: {training_time:.2f}s")
            
            return model, metrics
    
    def run_experiments(self, X_train, X_test, y_train, y_test):
        """
        Run multiple MLflow experiments with different algorithms and parameters
        
        Args:
            X_train, X_test, y_train, y_test: Data splits
            
        Returns:
            dict: Results of all experiments
        """
        logger.info("Starting MLflow experiments...")
        
        all_results = {}
        run_count = 0
        
        # Logistic Regression Experiments (4 runs)
        lr_params_list = [
            {'C': 0.1, 'solver': 'liblinear', 'max_iter': 1000, 'random_seed': 42},
            {'C': 1.0, 'solver': 'liblinear', 'max_iter': 1000, 'random_seed': 42},
            {'C': 10.0, 'solver': 'liblinear', 'max_iter': 1000, 'random_seed': 42},
            {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'random_seed': 42}
        ]
        
        for i, params in enumerate(lr_params_list):
            run_count += 1
            logger.info(f"Running Logistic Regression experiment {i+1}/4...")
            model, metrics = self.train_logistic_regression(X_train, X_test, y_train, y_test, params)
            
            # Track best model
            if metrics['accuracy'] > self.best_score:
                self.best_score = metrics['accuracy']
                self.best_model = model
                self.best_run_id = mlflow.active_run().info.run_id
            
            all_results[f"lr_{i+1}"] = {'model': model, 'metrics': metrics, 'params': params}
        
        # Naive Bayes Experiments (4 runs)
        nb_params_list = [
            {'alpha': 0.1, 'fit_prior': True, 'random_seed': 42},
            {'alpha': 0.5, 'fit_prior': True, 'random_seed': 42},
            {'alpha': 1.0, 'fit_prior': True, 'random_seed': 42},
            {'alpha': 1.0, 'fit_prior': False, 'random_seed': 42}
        ]
        
        for i, params in enumerate(nb_params_list):
            run_count += 1
            logger.info(f"Running Naive Bayes experiment {i+1}/4...")
            model, metrics = self.train_naive_bayes(X_train, X_test, y_train, y_test, params)
            
            # Track best model
            if metrics['accuracy'] > self.best_score:
                self.best_score = metrics['accuracy']
                self.best_model = model
                self.best_run_id = mlflow.active_run().info.run_id
            
            all_results[f"nb_{i+1}"] = {'model': model, 'metrics': metrics, 'params': params}
        
        # Random Forest Experiments (4 runs)
        rf_params_list = [
            {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_seed': 42},
            {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_seed': 42},
            {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_seed': 42},
            {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_seed': 42}
        ]
        
        for i, params in enumerate(rf_params_list):
            run_count += 1
            logger.info(f"Running Random Forest experiment {i+1}/4...")
            model, metrics = self.train_random_forest(X_train, X_test, y_train, y_test, params)
            
            # Track best model
            if metrics['accuracy'] > self.best_score:
                self.best_score = metrics['accuracy']
                self.best_model = model
                self.best_run_id = mlflow.active_run().info.run_id
            
            all_results[f"rf_{i+1}"] = {'model': model, 'metrics': metrics, 'params': params}
        
        logger.info(f"Completed {run_count} MLflow experiments!")
        logger.info(f"Best model accuracy: {self.best_score:.4f}")
        logger.info(f"Best run ID: {self.best_run_id}")
        
        return all_results
    
    def save_best_model(self, output_dir="../models"):
        """
        Save the best model
        
        Args:
            output_dir (str): Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the best model
        model_path = os.path.join(output_dir, 'best_model.pkl')
        joblib.dump(self.best_model, model_path)
        
        # Save model metadata
        metadata = {
            'best_score': self.best_score,
            'best_run_id': self.best_run_id,
            'experiment_name': self.experiment_name
        }
        
        metadata_path = os.path.join(output_dir, 'best_model_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Best model saved to {model_path}")
        logger.info(f"Model metadata saved to {metadata_path}")
    
    def print_experiment_summary(self, all_results):
        """
        Print summary of all experiments
        
        Args:
            all_results (dict): Results of all experiments
        """
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        for exp_name, result in all_results.items():
            metrics = result['metrics']
            params = result['params']
            
            print(f"\n{exp_name.upper()}:")
            print(f"  Parameters: {params}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  Training Time: {metrics['training_time_seconds']:.2f}s")
            print(f"  Model Size: {metrics['model_size_mb']:.2f} MB")
        
        print(f"\nBEST MODEL:")
        print(f"  Accuracy: {self.best_score:.4f}")
        print(f"  Run ID: {self.best_run_id}")
        print("="*80)

def main():
    """
    Main function to execute training pipeline
    """
    try:
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Load data
        X_train, X_test, y_train, y_test, label_encoder = trainer.load_data()
        
        # Run experiments
        all_results = trainer.run_experiments(X_train, X_test, y_train, y_test)
        
        # Save best model
        trainer.save_best_model()
        
        # Print summary
        trainer.print_experiment_summary(all_results)
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
