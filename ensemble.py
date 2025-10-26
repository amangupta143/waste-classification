import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from tqdm import tqdm
from collections import defaultdict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Warning: ultralytics not installed. YOLO models will be skipped.")


class WasteEnsembleModel:
    """
    Create and evaluate ensemble models for waste classification
    Implements voting, weighted voting, and stacking ensemble methods
    """
    
    def __init__(self, results_path='results', num_classes=9, class_names=None):
        """
        Initialize ensemble model
        
        Args:
            results_path: Path to results directory
            num_classes: Number of classification categories
            class_names: List of class names
        """
        self.results_path = Path(results_path)
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        
        self.models_path = self.results_path / 'models'
        self.plots_path = self.results_path / 'plots'
        self.metrics_path = self.results_path / 'metrics'
        
        # Storage for models and predictions
        self.keras_models = {}
        self.yolo_models = {}
        self.model_predictions = {}
        self.model_weights = {}
        
        print("="*60)
        print("ENSEMBLE MODEL CREATOR")
        print("="*60)
        print(f"Results path: {self.results_path}")
        print(f"Number of classes: {self.num_classes}")
    
    def load_keras_models(self, model_names=None):
        """
        Load trained Keras models
        
        Args:
            model_names: List of model names to load (None = load all)
        """
        print("\n" + "="*60)
        print("LOADING KERAS MODELS")
        print("="*60)
        
        if model_names is None:
            # Find all .h5 files in models directory
            model_files = list(self.models_path.glob('*_best.h5'))
            model_names = [f.stem.replace('_best', '') for f in model_files]
        
        loaded_count = 0
        for model_name in model_names:
            model_path = self.models_path / f'{model_name}_best.h5'
            
            if not model_path.exists():
                print(f"⚠️  Model not found: {model_path}")
                continue
            
            try:
                print(f"Loading {model_name}...", end=" ")
                model = keras.models.load_model(str(model_path))
                self.keras_models[model_name] = model
                loaded_count += 1
                print("✓")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        print(f"\n✓ Loaded {loaded_count} Keras models")
        return self.keras_models
    
    def load_yolo_models(self, model_names=None):
        """
        Load trained YOLO models
        
        Args:
            model_names: List of YOLO model names to load
        """
        if not YOLO_AVAILABLE:
            print("\n⚠️  YOLO not available, skipping YOLO models")
            return {}
        
        print("\n" + "="*60)
        print("LOADING YOLO MODELS")
        print("="*60)
        
        if model_names is None:
            # Find all YOLO .pt files
            yolo_files = list(self.models_path.glob('YOLO*_best.pt'))
            model_names = [f.stem.replace('_best', '') for f in yolo_files]
        
        loaded_count = 0
        for model_name in model_names:
            model_path = self.models_path / f'{model_name}_best.pt'
            
            if not model_path.exists():
                print(f"⚠️  Model not found: {model_path}")
                continue
            
            try:
                print(f"Loading {model_name}...", end=" ")
                model = YOLO(str(model_path))
                self.yolo_models[model_name] = model
                loaded_count += 1
                print("✓")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        print(f"\n✓ Loaded {loaded_count} YOLO models")
        return self.yolo_models
    
    def predict_keras_model(self, model, model_name, X, batch_size=32):
        """
        Generate predictions from a Keras model
        
        Args:
            model: Keras model
            model_name: Name of the model
            X: Input data (normalized)
            batch_size: Batch size for prediction
            
        Returns:
            Predictions array
        """
        print(f"Predicting with {model_name}...", end=" ")
        predictions = model.predict(X, batch_size=batch_size, verbose=0)
        print("✓")
        return predictions
    
    def predict_yolo_model(self, model, model_name, X_original):
        """
        Generate predictions from a YOLO model
        
        Args:
            model: YOLO model
            model_name: Name of the model
            X_original: Input data (UNNORMALIZED, 0-255 range)
            
        Returns:
            Predictions array (num_samples, num_classes)
        """
        print(f"Predicting with {model_name}...")
        predictions = []
        
        for idx in tqdm(range(len(X_original)), desc=f"  {model_name}"):
            img = X_original[idx].astype(np.uint8)
            
            # YOLO prediction
            results = model(img, verbose=False)
            probs = results[0].probs.data.cpu().numpy()
            predictions.append(probs)
        
        predictions = np.array(predictions)
        print("  ✓")
        return predictions
    
    def generate_all_predictions(self, X_test, X_test_original=None):
        """
        Generate predictions from all loaded models
        
        Args:
            X_test: Normalized test data for Keras models
            X_test_original: Original test data for YOLO models
            
        Returns:
            Dictionary of predictions
        """
        print("\n" + "="*60)
        print("GENERATING PREDICTIONS FROM ALL MODELS")
        print("="*60)
        
        # Keras models
        print(f"\nKeras models ({len(self.keras_models)}):")
        for model_name, model in self.keras_models.items():
            predictions = self.predict_keras_model(model, model_name, X_test)
            self.model_predictions[model_name] = predictions
        
        # YOLO models
        if self.yolo_models and X_test_original is not None:
            print(f"\nYOLO models ({len(self.yolo_models)}):")
            for model_name, model in self.yolo_models.items():
                predictions = self.predict_yolo_model(model, model_name, X_test_original)
                self.model_predictions[model_name] = predictions
        
        print(f"\n✓ Generated predictions from {len(self.model_predictions)} models")
        return self.model_predictions
    
    def calculate_model_weights(self, y_true):
        """
        Calculate optimal weights for each model based on validation accuracy
        
        Args:
            y_true: True labels
            
        Returns:
            Dictionary of model weights
        """
        print("\n" + "="*60)
        print("CALCULATING MODEL WEIGHTS")
        print("="*60)
        
        y_true_labels = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true
        
        accuracies = {}
        for model_name, predictions in self.model_predictions.items():
            y_pred = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(y_true_labels, y_pred)
            accuracies[model_name] = accuracy
            print(f"{model_name}: {accuracy:.4f}")
        
        # Normalize accuracies to get weights
        total_accuracy = sum(accuracies.values())
        self.model_weights = {
            name: acc / total_accuracy 
            for name, acc in accuracies.items()
        }
        
        print("\nNormalized weights:")
        for model_name, weight in self.model_weights.items():
            print(f"  {model_name}: {weight:.4f}")
        
        return self.model_weights
    
    def voting_ensemble(self, predictions_dict, method='hard'):
        """
        Create voting ensemble
        
        Args:
            predictions_dict: Dictionary of model predictions
            method: 'hard' for majority voting, 'soft' for average probabilities
            
        Returns:
            Ensemble predictions
        """
        if method == 'hard':
            # Hard voting: majority class
            votes = []
            for model_name, preds in predictions_dict.items():
                votes.append(np.argmax(preds, axis=1))
            
            votes = np.array(votes)
            # Get majority vote for each sample
            ensemble_preds = np.apply_along_axis(
                lambda x: np.bincount(x, minlength=self.num_classes).argmax(),
                axis=0,
                arr=votes
            )
            
        else:  # soft voting
            # Soft voting: average probabilities
            all_probs = np.array(list(predictions_dict.values()))
            avg_probs = np.mean(all_probs, axis=0)
            ensemble_preds = np.argmax(avg_probs, axis=1)
        
        return ensemble_preds
    
    def weighted_ensemble(self, predictions_dict, weights=None):
        """
        Create weighted ensemble
        
        Args:
            predictions_dict: Dictionary of model predictions
            weights: Dictionary of model weights (None = use calculated weights)
            
        Returns:
            Ensemble predictions
        """
        if weights is None:
            weights = self.model_weights
        
        # Weighted average of probabilities
        weighted_probs = np.zeros((len(list(predictions_dict.values())[0]), self.num_classes))
        
        for model_name, preds in predictions_dict.items():
            weight = weights.get(model_name, 1.0 / len(predictions_dict))
            weighted_probs += preds * weight
        
        ensemble_preds = np.argmax(weighted_probs, axis=1)
        
        return ensemble_preds
    
    def stacking_ensemble(self, train_predictions, y_train, test_predictions, 
                         meta_model='logistic'):
        """
        Create stacking ensemble with meta-learner
        
        Args:
            train_predictions: Dictionary of predictions on training/validation data
            y_train: True training labels
            test_predictions: Dictionary of predictions on test data
            meta_model: Type of meta-learner ('logistic' or 'random_forest')
            
        Returns:
            Ensemble predictions and trained meta-model
        """
        print("\n" + "="*60)
        print("TRAINING STACKING ENSEMBLE")
        print("="*60)
        
        # Prepare training data for meta-model
        X_meta_train = np.hstack([preds for preds in train_predictions.values()])
        y_train_labels = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train
        
        print(f"Meta-model input shape: {X_meta_train.shape}")
        print(f"Meta-model type: {meta_model}")
        
        # Train meta-model
        if meta_model == 'logistic':
            meta_learner = LogisticRegression(
                max_iter=1000,
                multi_class='multinomial',
                random_state=42
            )
        else:  # random_forest
            meta_learner = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        
        print("Training meta-model...", end=" ")
        meta_learner.fit(X_meta_train, y_train_labels)
        print("✓")
        
        # Prepare test data for meta-model
        X_meta_test = np.hstack([preds for preds in test_predictions.values()])
        
        # Predict
        print("Generating ensemble predictions...", end=" ")
        ensemble_preds = meta_learner.predict(X_meta_test)
        print("✓")
        
        return ensemble_preds, meta_learner
    
    def evaluate_ensemble(self, y_true, y_pred, ensemble_name):
        """
        Evaluate ensemble performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            ensemble_name: Name of the ensemble
            
        Returns:
            Dictionary of metrics
        """
        y_true_labels = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true
        
        metrics = {
            'ensemble_name': ensemble_name,
            'accuracy': accuracy_score(y_true_labels, y_pred),
            'precision_macro': precision_score(y_true_labels, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true_labels, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true_labels, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true_labels, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true_labels, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true_labels, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, ensemble_name):
        """
        Plot and save confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            ensemble_name: Name of the ensemble
        """
        y_true_labels = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true
        
        cm = confusion_matrix(y_true_labels, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title(f'{ensemble_name} - Confusion Matrix', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filename = f'{ensemble_name.replace(" ", "_")}_confusion_matrix.png'
        plt.savefig(self.plots_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Confusion matrix saved: {filename}")
    
    def create_classification_report(self, y_true, y_pred, ensemble_name):
        """
        Create and save detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            ensemble_name: Name of the ensemble
        """
        y_true_labels = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true
        
        report = classification_report(
            y_true_labels, y_pred,
            target_names=self.class_names,
            digits=4,
            output_dict=True
        )
        
        # Convert to DataFrame for better visualization
        df = pd.DataFrame(report).transpose()
        
        # Save to CSV
        filename = f'{ensemble_name.replace(" ", "_")}_classification_report.csv'
        df.to_csv(self.metrics_path / filename)
        
        print(f"  ✓ Classification report saved: {filename}")
        
        return df
    
    def compare_ensembles(self, ensemble_results):
        """
        Compare performance of different ensemble methods
        
        Args:
            ensemble_results: Dictionary of ensemble metrics
        """
        print("\n" + "="*60)
        print("ENSEMBLE COMPARISON")
        print("="*60)
        
        # Create comparison DataFrame
        df = pd.DataFrame(ensemble_results)
        df = df.sort_values('accuracy', ascending=False)
        
        print("\n" + df.to_string(index=False))
        
        # Save to CSV
        df.to_csv(self.metrics_path / 'ensemble_comparison.csv', index=False)
        print(f"\n✓ Comparison saved: ensemble_comparison.csv")
        
        # Plot comparison
        self.plot_ensemble_comparison(df)
        
        return df
    
    def plot_ensemble_comparison(self, comparison_df):
        """
        Create visual comparison of ensemble methods
        
        Args:
            comparison_df: DataFrame with ensemble metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ensembles = comparison_df['ensemble_name'].values
        x_pos = np.arange(len(ensembles))
        
        metrics_to_plot = [
            ('accuracy', 'Accuracy', 'skyblue'),
            ('f1_macro', 'F1-Score (Macro)', 'lightcoral'),
            ('precision_macro', 'Precision (Macro)', 'lightgreen'),
            ('recall_macro', 'Recall (Macro)', 'plum')
        ]
        
        for idx, (metric, title, color) in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            values = comparison_df[metric].values
            
            bars = ax.barh(x_pos, values, color=color, edgecolor='navy', linewidth=1.5)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val + 0.01, i, f'{val:.4f}', 
                       va='center', fontsize=10, fontweight='bold')
            
            ax.set_yticks(x_pos)
            ax.set_yticklabels(ensembles, fontsize=10)
            ax.set_xlabel(title, fontsize=11, fontweight='bold')
            ax.set_title(f'{title} Comparison', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1.1)
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'ensemble_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Comparison plot saved: ensemble_comparison.png")
    
    def save_ensemble_model(self, ensemble_type, ensemble_data):
        """
        Save ensemble model configuration and weights
        
        Args:
            ensemble_type: Type of ensemble
            ensemble_data: Data to save (weights, meta-model, etc.)
        """
        ensemble_path = self.models_path / 'ensembles'
        ensemble_path.mkdir(exist_ok=True)
        
        if ensemble_type == 'weighted':
            # Save weights
            with open(ensemble_path / 'weighted_ensemble_weights.json', 'w') as f:
                json.dump(ensemble_data, f, indent=4)
            print(f"  ✓ Weighted ensemble weights saved")
        
        elif ensemble_type == 'stacking':
            # Save meta-model
            import joblib
            meta_model = ensemble_data['meta_model']
            joblib.dump(meta_model, ensemble_path / 'stacking_meta_model.pkl')
            
            # Save configuration
            config = {
                'model_names': list(self.model_predictions.keys()),
                'meta_model_type': type(meta_model).__name__
            }
            with open(ensemble_path / 'stacking_config.json', 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"  ✓ Stacking ensemble meta-model saved")
    
    def run_all_ensembles(self, X_val, y_val, X_test, y_test, 
                         X_val_original=None, X_test_original=None):
        """
        Run all ensemble methods and compare results
        
        Args:
            X_val: Validation data (normalized)
            y_val: Validation labels
            X_test: Test data (normalized)
            y_test: Test labels
            X_val_original: Original validation data for YOLO
            X_test_original: Original test data for YOLO
            
        Returns:
            Dictionary of all ensemble results
        """
        print("\n" + "="*60)
        print("RUNNING ALL ENSEMBLE METHODS")
        print("="*60)
        
        # Generate predictions for validation set (ALL MODELS including YOLO)
        print("\n📊 Generating validation predictions...")
        val_predictions = {}
        
        # Keras models
        for model_name, model in self.keras_models.items():
            val_predictions[model_name] = self.predict_keras_model(
                model, model_name, X_val
            )
        
        # YOLO models (if available)
        if self.yolo_models and X_val_original is not None:
            print(f"\nYOLO validation predictions ({len(self.yolo_models)}):")
            for model_name, model in self.yolo_models.items():
                val_predictions[model_name] = self.predict_yolo_model(
                    model, model_name, X_val_original
                )
        
        # Store validation predictions for stacking
        self.val_predictions = val_predictions
        
        # Calculate model weights based on validation performance
        # Use validation predictions for weight calculation
        temp_predictions = self.model_predictions
        self.model_predictions = val_predictions
        self.calculate_model_weights(y_val)
        self.model_predictions = temp_predictions  # Restore
        
        # Generate test predictions (ALL MODELS including YOLO)
        test_predictions = self.generate_all_predictions(X_test, X_test_original)
        
        # Ensure both val and test have the same models
        common_models = set(val_predictions.keys()) & set(test_predictions.keys())
        if len(common_models) < len(val_predictions) or len(common_models) < len(test_predictions):
            print(f"\n⚠️  Warning: Different models in validation and test sets")
            print(f"   Validation models: {list(val_predictions.keys())}")
            print(f"   Test models: {list(test_predictions.keys())}")
            print(f"   Using common models: {list(common_models)}")
            
            val_predictions = {k: v for k, v in val_predictions.items() if k in common_models}
            test_predictions = {k: v for k, v in test_predictions.items() if k in common_models}
        
        # Run ensemble methods
        ensemble_results = []
        
        # 1. Hard Voting Ensemble
        print("\n" + "-"*60)
        print("1. HARD VOTING ENSEMBLE")
        print("-"*60)
        y_pred_hard = self.voting_ensemble(test_predictions, method='hard')
        metrics_hard = self.evaluate_ensemble(y_test, y_pred_hard, 'Hard Voting')
        ensemble_results.append(metrics_hard)
        print(f"Accuracy: {metrics_hard['accuracy']:.4f}")
        self.plot_confusion_matrix(y_test, y_pred_hard, 'Hard Voting Ensemble')
        self.create_classification_report(y_test, y_pred_hard, 'Hard Voting Ensemble')
        
        # 2. Soft Voting Ensemble
        print("\n" + "-"*60)
        print("2. SOFT VOTING ENSEMBLE")
        print("-"*60)
        y_pred_soft = self.voting_ensemble(test_predictions, method='soft')
        metrics_soft = self.evaluate_ensemble(y_test, y_pred_soft, 'Soft Voting')
        ensemble_results.append(metrics_soft)
        print(f"Accuracy: {metrics_soft['accuracy']:.4f}")
        self.plot_confusion_matrix(y_test, y_pred_soft, 'Soft Voting Ensemble')
        self.create_classification_report(y_test, y_pred_soft, 'Soft Voting Ensemble')
        
        # 3. Weighted Ensemble
        print("\n" + "-"*60)
        print("3. WEIGHTED ENSEMBLE")
        print("-"*60)
        y_pred_weighted = self.weighted_ensemble(test_predictions, self.model_weights)
        metrics_weighted = self.evaluate_ensemble(y_test, y_pred_weighted, 'Weighted Voting')
        ensemble_results.append(metrics_weighted)
        print(f"Accuracy: {metrics_weighted['accuracy']:.4f}")
        self.plot_confusion_matrix(y_test, y_pred_weighted, 'Weighted Voting Ensemble')
        self.create_classification_report(y_test, y_pred_weighted, 'Weighted Voting Ensemble')
        self.save_ensemble_model('weighted', self.model_weights)
        
        # 4. Stacking Ensemble (Logistic Regression)
        print("\n" + "-"*60)
        print("4. STACKING ENSEMBLE (Logistic Regression)")
        print("-"*60)
        print(f"Using {len(val_predictions)} models for stacking")
        y_pred_stack_lr, meta_model_lr = self.stacking_ensemble(
            val_predictions, y_val, test_predictions, meta_model='logistic'
        )
        metrics_stack_lr = self.evaluate_ensemble(y_test, y_pred_stack_lr, 
                                                  'Stacking (Logistic)')
        ensemble_results.append(metrics_stack_lr)
        print(f"Accuracy: {metrics_stack_lr['accuracy']:.4f}")
        self.plot_confusion_matrix(y_test, y_pred_stack_lr, 'Stacking Ensemble (Logistic)')
        self.create_classification_report(y_test, y_pred_stack_lr, 'Stacking Ensemble (Logistic)')
        self.save_ensemble_model('stacking', {'meta_model': meta_model_lr})
        
        # 5. Stacking Ensemble (Random Forest)
        print("\n" + "-"*60)
        print("5. STACKING ENSEMBLE (Random Forest)")
        print("-"*60)
        y_pred_stack_rf, meta_model_rf = self.stacking_ensemble(
            val_predictions, y_val, test_predictions, meta_model='random_forest'
        )
        metrics_stack_rf = self.evaluate_ensemble(y_test, y_pred_stack_rf, 
                                                  'Stacking (Random Forest)')
        ensemble_results.append(metrics_stack_rf)
        print(f"Accuracy: {metrics_stack_rf['accuracy']:.4f}")
        self.plot_confusion_matrix(y_test, y_pred_stack_rf, 'Stacking Ensemble (Random Forest)')
        self.create_classification_report(y_test, y_pred_stack_rf, 'Stacking Ensemble (Random Forest)')
        
        # Compare all ensembles
        comparison_df = self.compare_ensembles(ensemble_results)
        
        # Add individual model performance for comparison
        print("\n" + "="*60)
        print("INDIVIDUAL MODEL PERFORMANCE")
        print("="*60)
        individual_results = []
        for model_name, predictions in test_predictions.items():
            y_pred = np.argmax(predictions, axis=1)
            metrics = self.evaluate_ensemble(y_test, y_pred, model_name)
            individual_results.append(metrics)
            print(f"{model_name}: {metrics['accuracy']:.4f}")
        
        # Save individual results
        individual_df = pd.DataFrame(individual_results)
        individual_df = individual_df.sort_values('accuracy', ascending=False)
        individual_df.to_csv(self.metrics_path / 'individual_model_performance.csv', 
                           index=False)
        
        # Create final comparison including best individual model
        print("\n" + "="*60)
        print("BEST RESULTS COMPARISON")
        print("="*60)
        
        best_individual = individual_df.iloc[0]
        best_ensemble = comparison_df.iloc[0]
        
        print(f"\n🏆 Best Individual Model: {best_individual['ensemble_name']}")
        print(f"   Accuracy: {best_individual['accuracy']:.4f}")
        print(f"   F1-Score: {best_individual['f1_macro']:.4f}")
        
        print(f"\n🏆 Best Ensemble Model: {best_ensemble['ensemble_name']}")
        print(f"   Accuracy: {best_ensemble['accuracy']:.4f}")
        print(f"   F1-Score: {best_ensemble['f1_macro']:.4f}")
        
        improvement = (best_ensemble['accuracy'] - best_individual['accuracy']) * 100
        print(f"\n📈 Ensemble Improvement: {improvement:+.2f}%")
        
        return {
            'ensemble_results': ensemble_results,
            'individual_results': individual_results,
            'comparison_df': comparison_df,
            'individual_df': individual_df,
            'best_ensemble': best_ensemble,
            'best_individual': best_individual
        }


def main():
    """Main execution function"""
    print("="*60)
    print("WASTE CLASSIFICATION - ENSEMBLE MODELS")
    print("="*60)
    
    # Check if data and models exist
    results_path = Path('results')
    processed_data_path = results_path / 'processed_data'
    models_path = results_path / 'models'
    
    if not processed_data_path.exists():
        print("\n❌ Error: Processed data not found!")
        print("Please run data_preparation.py first.")
        return
    
    if not models_path.exists() or not list(models_path.glob('*_best.h5')):
        print("\n❌ Error: No trained models found!")
        print("Please run train_models.py first.")
        return
    
    # Load data
    print("\nLoading preprocessed data...")
    X_train = np.load(processed_data_path / 'X_train.npy')
    X_val = np.load(processed_data_path / 'X_val.npy')
    X_test = np.load(processed_data_path / 'X_test.npy')
    y_train = np.load(processed_data_path / 'y_train.npy')
    y_val = np.load(processed_data_path / 'y_val.npy')
    y_test = np.load(processed_data_path / 'y_test.npy')
    
    print(f"✓ Training data: {X_train.shape}")
    print(f"✓ Validation data: {X_val.shape}")
    print(f"✓ Test data: {X_test.shape}")
    
    # Normalize data for Keras models
    X_val_normalized = X_val / 255.0
    X_test_normalized = X_test / 255.0
    
    # Load data split info
    with open(results_path / 'metrics' / 'data_split_info.json', 'r') as f:
        data_info = json.load(f)
    
    class_names = data_info['class_names']
    num_classes = data_info['num_classes']
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    # Initialize ensemble model
    ensemble = WasteEnsembleModel(
        results_path='results',
        num_classes=num_classes,
        class_names=class_names
    )
    
    # Load trained models
    print("\n" + "="*60)
    print("MODEL LOADING")
    print("="*60)
    
    # Find available Keras models
    keras_model_files = list(models_path.glob('*_best.h5'))
    keras_model_names = [f.stem.replace('_best', '') for f in keras_model_files]
    
    # Find available YOLO models
    yolo_model_files = list(models_path.glob('YOLO*_best.pt'))
    yolo_model_names = [f.stem.replace('_best', '') for f in yolo_model_files]
    
    print(f"\nAvailable Keras models: {len(keras_model_names)}")
    for name in keras_model_names:
        print(f"  - {name}")
    
    if yolo_model_names:
        print(f"\nAvailable YOLO models: {len(yolo_model_names)}")
        for name in yolo_model_names:
            print(f"  - {name}")
    
    # Ask user which models to include in ensemble
    print("\n" + "="*60)
    print("MODEL SELECTION FOR ENSEMBLE")
    print("="*60)
    print("\nOptions:")
    print("  1. Use all available models (recommended)")
    print("  2. Use only Keras models")
    print("  3. Use top 5 best performing models")
    print("  4. Custom selection")
    
    choice = input("\nSelect option (1-4) [default: 1]: ").strip() or "1"
    
    models_to_use_keras = []
    models_to_use_yolo = []
    
    if choice == "1":
        # Use all models
        models_to_use_keras = keras_model_names
        models_to_use_yolo = yolo_model_names
        print(f"\n✓ Using all {len(keras_model_names) + len(yolo_model_names)} models")
    
    elif choice == "2":
        # Use only Keras models
        models_to_use_keras = keras_model_names
        print(f"\n✓ Using {len(keras_model_names)} Keras models")
    
    elif choice == "3":
        # Use top 5 models based on training summary
        summary_file = results_path / 'metrics' / 'training_summary.csv'
        if summary_file.exists():
            summary_df = pd.read_csv(summary_file)
            top_models = summary_df.nlargest(5, 'Best Val Accuracy')['Model'].tolist()
            
            models_to_use_keras = [m for m in top_models if m in keras_model_names]
            models_to_use_yolo = [m for m in top_models if m in yolo_model_names]
            
            print(f"\n✓ Using top 5 models:")
            for model in top_models:
                print(f"  - {model}")
        else:
            print("\n⚠️  Training summary not found, using all models")
            models_to_use_keras = keras_model_names
            models_to_use_yolo = yolo_model_names
    
    else:
        # Custom selection
        print("\nEnter model names separated by commas:")
        custom_input = input("Models: ").strip()
        selected = [m.strip() for m in custom_input.split(',')]
        
        models_to_use_keras = [m for m in selected if m in keras_model_names]
        models_to_use_yolo = [m for m in selected if m in yolo_model_names]
        
        print(f"\n✓ Selected {len(models_to_use_keras) + len(models_to_use_yolo)} models")
    
    # Load models
    ensemble.load_keras_models(models_to_use_keras)
    
    if models_to_use_yolo and YOLO_AVAILABLE:
        ensemble.load_yolo_models(models_to_use_yolo)
    
    total_models = len(ensemble.keras_models) + len(ensemble.yolo_models)
    
    if total_models < 2:
        print("\n❌ Error: Need at least 2 models for ensemble!")
        print("Please train more models first.")
        return
    
    print(f"\n✓ Total models loaded: {total_models}")
    
    # Run all ensemble methods
    print("\n" + "="*60)
    print("ENSEMBLE CREATION AND EVALUATION")
    print("="*60)
    print("\nThis will:")
    print("  1. Generate predictions from all loaded models")
    print("  2. Calculate optimal model weights")
    print("  3. Create multiple ensemble models:")
    print("     - Hard Voting Ensemble")
    print("     - Soft Voting Ensemble")
    print("     - Weighted Ensemble")
    print("     - Stacking Ensemble (Logistic Regression)")
    print("     - Stacking Ensemble (Random Forest)")
    print("  4. Evaluate and compare all ensembles")
    print("  5. Generate visualizations and reports")
    
    response = input("\nProceed with ensemble creation? (yes/no) [yes]: ").strip().lower()
    if response and response not in ['yes', 'y']:
        print("Ensemble creation cancelled.")
        return
    
    # Run all ensembles
    start_time = time.time()
    
    results = ensemble.run_all_ensembles(
        X_val_normalized, y_val,
        X_test_normalized, y_test,
        X_val_original=X_val,  # For YOLO models on validation
        X_test_original=X_test  # For YOLO models on test
    )
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "="*60)
    print("ENSEMBLE CREATION COMPLETE!")
    print("="*60)
    
    print(f"\n✓ Total time: {total_time/60:.2f} minutes")
    print(f"✓ Models used: {total_models}")
    print(f"✓ Ensemble methods evaluated: 5")
    
    print("\n📊 BEST RESULTS:")
    print("-"*60)
    
    best_ensemble = results['best_ensemble']
    best_individual = results['best_individual']
    
    print(f"\n🥇 Best Ensemble: {best_ensemble['ensemble_name']}")
    print(f"   Accuracy: {best_ensemble['accuracy']:.4f}")
    print(f"   Precision: {best_ensemble['precision_macro']:.4f}")
    print(f"   Recall: {best_ensemble['recall_macro']:.4f}")
    print(f"   F1-Score: {best_ensemble['f1_macro']:.4f}")
    
    print(f"\n🥈 Best Individual Model: {best_individual['ensemble_name']}")
    print(f"   Accuracy: {best_individual['accuracy']:.4f}")
    print(f"   Precision: {best_individual['precision_macro']:.4f}")
    print(f"   Recall: {best_individual['recall_macro']:.4f}")
    print(f"   F1-Score: {best_individual['f1_macro']:.4f}")
    
    improvement = (best_ensemble['accuracy'] - best_individual['accuracy']) * 100
    print(f"\n📈 Improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print("\n✅ Ensemble outperformed individual models!")
    elif improvement < -1:
        print("\n⚠️  Best individual model outperformed ensemble")
        print("   Consider using the individual model instead")
    else:
        print("\n➖ Ensemble and individual models perform similarly")
    
    # Save final summary
    summary = {
        'total_models_used': total_models,
        'keras_models': list(ensemble.keras_models.keys()),
        'yolo_models': list(ensemble.yolo_models.keys()),
        'ensemble_methods_evaluated': 5,
        'best_ensemble': {
            'name': best_ensemble['ensemble_name'],
            'accuracy': float(best_ensemble['accuracy']),
            'f1_score': float(best_ensemble['f1_macro'])
        },
        'best_individual': {
            'name': best_individual['ensemble_name'],
            'accuracy': float(best_individual['accuracy']),
            'f1_score': float(best_individual['f1_macro'])
        },
        'improvement_percentage': float(improvement),
        'total_time_seconds': total_time
    }
    
    with open(results_path / 'metrics' / 'ensemble_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\n" + "="*60)
    print("RESULTS SAVED")
    print("="*60)
    print(f"\n✓ All results saved to: {results_path}")
    print("\nGenerated files:")
    print("  📊 Metrics:")
    print("     - ensemble_comparison.csv")
    print("     - individual_model_performance.csv")
    print("     - ensemble_summary.json")
    print("     - *_classification_report.csv (for each ensemble)")
    print("\n  📈 Plots:")
    print("     - ensemble_comparison.png")
    print("     - *_confusion_matrix.png (for each ensemble)")
    print("\n  🤖 Models:")
    print("     - ensembles/weighted_ensemble_weights.json")
    print("     - ensembles/stacking_meta_model.pkl")
    print("     - ensembles/stacking_config.json")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if improvement > 2:
        print("\n✅ Use ensemble model for deployment")
        print(f"   Recommended: {best_ensemble['ensemble_name']}")
    elif improvement > 0:
        print("\n✅ Ensemble provides slight improvement")
        print(f"   Recommended: {best_ensemble['ensemble_name']}")
        print("   Alternative: Best individual model for faster inference")
    else:
        print("\n✅ Use best individual model")
        print(f"   Recommended: {best_individual['ensemble_name']}")
        print("   Ensemble doesn't provide significant benefit")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Review confusion matrices and classification reports")
    print("2. Run evaluate.py for detailed per-class analysis")
    print("3. Run predict.py to test on new images")
    print("4. Export best model for deployment")
    
    # Create final recommendation report
    create_recommendation_report(results, summary, results_path)
    
    print("\n✓ Recommendation report created: ensemble_recommendations.txt")
    print("\n" + "="*60)
    print("ENSEMBLE PIPELINE COMPLETE!")
    print("="*60)


def create_recommendation_report(results, summary, results_path):
    """
    Create a detailed recommendation report
    
    Args:
        results: Dictionary of ensemble results
        summary: Summary dictionary
        results_path: Path to save report
    """
    report_path = results_path / 'reports' / 'ensemble_recommendations.txt'
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("WASTE CLASSIFICATION - ENSEMBLE MODEL RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Models Evaluated: {summary['total_models_used']}\n")
        f.write(f"Ensemble Methods Tested: {summary['ensemble_methods_evaluated']}\n")
        f.write(f"Processing Time: {summary['total_time_seconds']/60:.2f} minutes\n\n")
        
        f.write("BEST ENSEMBLE MODEL\n")
        f.write("-"*80 + "\n")
        best_ens = summary['best_ensemble']
        f.write(f"Method: {best_ens['name']}\n")
        f.write(f"Accuracy: {best_ens['accuracy']:.4f} ({best_ens['accuracy']*100:.2f}%)\n")
        f.write(f"F1-Score: {best_ens['f1_score']:.4f}\n\n")
        
        f.write("BEST INDIVIDUAL MODEL\n")
        f.write("-"*80 + "\n")
        best_ind = summary['best_individual']
        f.write(f"Model: {best_ind['name']}\n")
        f.write(f"Accuracy: {best_ind['accuracy']:.4f} ({best_ind['accuracy']*100:.2f}%)\n")
        f.write(f"F1-Score: {best_ind['f1_score']:.4f}\n\n")
        
        f.write("PERFORMANCE IMPROVEMENT\n")
        f.write("-"*80 + "\n")
        improvement = summary['improvement_percentage']
        f.write(f"Ensemble vs Individual: {improvement:+.2f}%\n\n")
        
        f.write("DETAILED RANKINGS\n")
        f.write("-"*80 + "\n")
        f.write("\nEnsemble Methods (by accuracy):\n")
        for idx, row in results['comparison_df'].iterrows():
            f.write(f"  {idx+1}. {row['ensemble_name']}: {row['accuracy']:.4f}\n")
        
        f.write("\nIndividual Models (top 5 by accuracy):\n")
        for idx, row in results['individual_df'].head(5).iterrows():
            f.write(f"  {idx+1}. {row['ensemble_name']}: {row['accuracy']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DEPLOYMENT RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        if improvement > 2:
            f.write("RECOMMENDATION: Deploy Ensemble Model\n\n")
            f.write(f"The {best_ens['name']} provides a significant {improvement:.2f}% \n")
            f.write("improvement over individual models. This justifies the added \n")
            f.write("computational complexity.\n\n")
            f.write("Deployment Options:\n")
            f.write(f"  1. Primary: {best_ens['name']}\n")
            f.write(f"  2. Fallback: {best_ind['name']} (for faster inference)\n")
        
        elif improvement > 0:
            f.write("RECOMMENDATION: Consider Ensemble or Best Individual\n\n")
            f.write(f"The ensemble provides a {improvement:.2f}% improvement, which is modest.\n")
            f.write("Choose based on your priorities:\n\n")
            f.write("  - If accuracy is critical: Use ensemble\n")
            f.write("  - If speed is important: Use best individual model\n\n")
            f.write("Deployment Options:\n")
            f.write(f"  1. High Accuracy: {best_ens['name']}\n")
            f.write(f"  2. High Speed: {best_ind['name']}\n")
        
        else:
            f.write("RECOMMENDATION: Deploy Best Individual Model\n\n")
            f.write(f"The best individual model ({best_ind['name']}) performs as well \n")
            f.write("or better than ensemble methods, with faster inference time.\n\n")
            f.write("Deployment Options:\n")
            f.write(f"  1. Primary: {best_ind['name']}\n")
            f.write("  2. Backup: Train ensemble if more data becomes available\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TECHNICAL NOTES\n")
        f.write("="*80 + "\n\n")
        
        f.write("Models Used in Ensemble:\n")
        f.write("  Keras Models:\n")
        for model in summary['keras_models']:
            f.write(f"    - {model}\n")
        
        if summary['yolo_models']:
            f.write("  YOLO Models:\n")
            for model in summary['yolo_models']:
                f.write(f"    - {model}\n")
        
        f.write("\nEnsemble Methods Evaluated:\n")
        f.write("  1. Hard Voting: Majority class prediction\n")
        f.write("  2. Soft Voting: Average probability prediction\n")
        f.write("  3. Weighted Voting: Performance-weighted probabilities\n")
        f.write("  4. Stacking (Logistic): Meta-learner with logistic regression\n")
        f.write("  5. Stacking (Random Forest): Meta-learner with random forest\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")


if __name__ == "__main__":
    main()