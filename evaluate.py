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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Warning: ultralytics not installed. YOLO models will be skipped.")


class WasteModelEvaluator:
    """
    Comprehensive evaluation and analysis of waste classification models
    Includes per-class analysis, error analysis, and detailed visualizations
    """
    
    def __init__(self, results_path='results', num_classes=9, class_names=None):
        """
        Initialize model evaluator
        
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
        self.reports_path = self.results_path / 'reports'
        
        # Create evaluation subdirectories
        self.eval_plots_path = self.plots_path / 'evaluation'
        self.eval_metrics_path = self.metrics_path / 'evaluation'
        
        for path in [self.eval_plots_path, self.eval_metrics_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Storage for evaluation results
        self.evaluation_results = {}
        
        print("="*60)
        print("COMPREHENSIVE MODEL EVALUATOR")
        print("="*60)
        print(f"Results path: {self.results_path}")
        print(f"Number of classes: {self.num_classes}")
    
    def load_model(self, model_path, model_type='keras'):
        """
        Load a trained model
        
        Args:
            model_path: Path to model file
            model_type: 'keras' or 'yolo'
            
        Returns:
            Loaded model
        """
        if model_type == 'keras':
            return keras.models.load_model(str(model_path))
        elif model_type == 'yolo' and YOLO_AVAILABLE:
            return YOLO(str(model_path))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def predict_keras(self, model, X, batch_size=32):
        """Generate predictions from Keras model"""
        return model.predict(X, batch_size=batch_size, verbose=0)
    
    def predict_yolo(self, model, X_original):
        """Generate predictions from YOLO model"""
        predictions = []
        for idx in tqdm(range(len(X_original)), desc="Predicting"):
            img = X_original[idx].astype(np.uint8)
            results = model(img, verbose=False)
            probs = results[0].probs.data.cpu().numpy()
            predictions.append(probs)
        return np.array(predictions)
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        y_true_labels = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true
        
        metrics = {
            'accuracy': accuracy_score(y_true_labels, y_pred),
            'precision_macro': precision_score(y_true_labels, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true_labels, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true_labels, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true_labels, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true_labels, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true_labels, y_pred, average='weighted', zero_division=0)
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true_labels, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true_labels, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true_labels, y_pred, average=None, zero_division=0)
        
        metrics['per_class'] = {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1': f1_per_class.tolist()
        }
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true_labels, y_pred).tolist()
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, normalize=False):
        """
        Plot detailed confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            normalize: Whether to normalize the confusion matrix
        """
        y_true_labels = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true
        
        cm = confusion_matrix(y_true_labels, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title_suffix = '(Normalized)'
        else:
            fmt = 'd'
            title_suffix = '(Counts)'
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                   linewidths=0.5, linecolor='gray')
        
        plt.title(f'{model_name} - Confusion Matrix {title_suffix}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        suffix = '_normalized' if normalize else '_counts'
        filename = f'{model_name.replace(" ", "_")}_confusion_matrix{suffix}.png'
        plt.savefig(self.eval_plots_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm
    
    def plot_per_class_metrics(self, metrics, model_name):
        """
        Plot per-class performance metrics
        
        Args:
            metrics: Dictionary containing per-class metrics
            model_name: Name of the model
        """
        per_class = metrics['per_class']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        x_pos = np.arange(self.num_classes)
        
        metrics_to_plot = [
            ('precision', 'Precision', 'skyblue'),
            ('recall', 'Recall', 'lightcoral'),
            ('f1', 'F1-Score', 'lightgreen')
        ]
        
        for idx, (metric_name, title, color) in enumerate(metrics_to_plot):
            values = per_class[metric_name]
            
            bars = axes[idx].bar(x_pos, values, color=color, edgecolor='navy', linewidth=1.5)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{val:.3f}',
                             ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            axes[idx].set_xlabel('Class', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel(title, fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{title} per Class', fontsize=12, fontweight='bold')
            axes[idx].set_xticks(x_pos)
            axes[idx].set_xticklabels(self.class_names, rotation=45, ha='right')
            axes[idx].set_ylim(0, 1.1)
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].axhline(y=np.mean(values), color='red', linestyle='--', 
                            linewidth=2, label=f'Mean: {np.mean(values):.3f}')
            axes[idx].legend()
        
        plt.suptitle(f'{model_name} - Per-Class Performance', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filename = f'{model_name.replace(" ", "_")}_per_class_metrics.png'
        plt.savefig(self.eval_plots_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_misclassifications(self, y_true, y_pred, model_name):
        """
        Analyze and visualize misclassifications
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            DataFrame with misclassification analysis
        """
        y_true_labels = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true
        
        # Find misclassifications
        misclassified = y_true_labels != y_pred
        
        # Create misclassification matrix
        misclass_matrix = np.zeros((self.num_classes, self.num_classes))
        
        for true_label, pred_label in zip(y_true_labels[misclassified], 
                                          y_pred[misclassified]):
            misclass_matrix[true_label, pred_label] += 1
        
        # Plot misclassification matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(misclass_matrix, annot=True, fmt='.0f', cmap='Reds',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'{model_name} - Misclassification Pattern\n(True Label → Predicted Label)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filename = f'{model_name.replace(" ", "_")}_misclassification_matrix.png'
        plt.savefig(self.eval_plots_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed misclassification report
        misclass_data = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and misclass_matrix[i, j] > 0:
                    misclass_data.append({
                        'True Class': self.class_names[i],
                        'Predicted Class': self.class_names[j],
                        'Count': int(misclass_matrix[i, j]),
                        'Percentage of True Class': misclass_matrix[i, j] / np.sum(y_true_labels == i) * 100
                    })
        
        misclass_df = pd.DataFrame(misclass_data)
        if len(misclass_df) > 0:
            misclass_df = misclass_df.sort_values('Count', ascending=False)
        
        # Save report
        filename = f'{model_name.replace(" ", "_")}_misclassification_report.csv'
        misclass_df.to_csv(self.eval_metrics_path / filename, index=False)
        
        return misclass_df
    
    def plot_confidence_distribution(self, y_pred_proba, y_true, y_pred, model_name):
        """
        Plot confidence distribution for correct and incorrect predictions
        
        Args:
            y_pred_proba: Predicted probabilities
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
        """
        y_true_labels = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true
        
        # Get confidence scores (max probability)
        confidences = np.max(y_pred_proba, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = y_true_labels == y_pred
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        # Plot distributions
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        axes[0].hist(correct_confidences, bins=30, alpha=0.6, label='Correct', 
                    color='green', edgecolor='black')
        axes[0].hist(incorrect_confidences, bins=30, alpha=0.6, label='Incorrect', 
                    color='red', edgecolor='black')
        axes[0].set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Confidence Distribution', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(alpha=0.3)
        
        # Box plot
        data_to_plot = [correct_confidences, incorrect_confidences]
        bp = axes[1].boxplot(data_to_plot, labels=['Correct', 'Incorrect'],
                            patch_artist=True, showmeans=True)
        
        colors = ['lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[1].set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        axes[1].set_title('Confidence Comparison', fontsize=13, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add statistics
        stats_text = f"Correct:\nMean: {np.mean(correct_confidences):.3f}\nMedian: {np.median(correct_confidences):.3f}\n\n"
        stats_text += f"Incorrect:\nMean: {np.mean(incorrect_confidences):.3f}\nMedian: {np.median(incorrect_confidences):.3f}"
        axes[1].text(1.5, 0.5, stats_text, fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'{model_name} - Prediction Confidence Analysis', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{model_name.replace(" ", "_")}_confidence_distribution.png'
        plt.savefig(self.eval_plots_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return statistics
        return {
            'correct_mean': float(np.mean(correct_confidences)),
            'correct_median': float(np.median(correct_confidences)),
            'incorrect_mean': float(np.mean(incorrect_confidences)),
            'incorrect_median': float(np.median(incorrect_confidences))
        }
    
    def create_detailed_report(self, model_name, metrics, confidence_stats, 
                              misclass_df, y_true, y_pred):
        """
        Create comprehensive evaluation report
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
            confidence_stats: Confidence statistics
            misclass_df: Misclassification DataFrame
            y_true: True labels
            y_pred: Predicted labels
        """
        y_true_labels = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true
        
        report_path = self.reports_path / f'{model_name.replace(" ", "_")}_detailed_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"DETAILED EVALUATION REPORT: {model_name}\n")
            f.write("="*80 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"Precision (Macro): {metrics['precision_macro']:.4f}\n")
            f.write(f"Precision (Weighted): {metrics['precision_weighted']:.4f}\n")
            f.write(f"Recall (Macro): {metrics['recall_macro']:.4f}\n")
            f.write(f"Recall (Weighted): {metrics['recall_weighted']:.4f}\n")
            f.write(f"F1-Score (Macro): {metrics['f1_macro']:.4f}\n")
            f.write(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}\n\n")
            
            # Per-class performance
            f.write("PER-CLASS PERFORMANCE\n")
            f.write("-"*80 + "\n")
            per_class = metrics['per_class']
            
            for i, class_name in enumerate(self.class_names):
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {per_class['precision'][i]:.4f}\n")
                f.write(f"  Recall: {per_class['recall'][i]:.4f}\n")
                f.write(f"  F1-Score: {per_class['f1'][i]:.4f}\n")
                
                # Class distribution
                class_count = np.sum(y_true_labels == i)
                correct_count = np.sum((y_true_labels == i) & (y_pred == i))
                f.write(f"  Total Samples: {class_count}\n")
                f.write(f"  Correct Predictions: {correct_count} ({correct_count/class_count*100:.1f}%)\n")
            
            # Best and worst performing classes
            f.write("\n" + "-"*80 + "\n")
            f1_scores = per_class['f1']
            best_class_idx = np.argmax(f1_scores)
            worst_class_idx = np.argmin(f1_scores)
            
            f.write(f"\n🏆 Best Performing Class: {self.class_names[best_class_idx]}\n")
            f.write(f"   F1-Score: {f1_scores[best_class_idx]:.4f}\n")
            f.write(f"\n⚠️  Worst Performing Class: {self.class_names[worst_class_idx]}\n")
            f.write(f"   F1-Score: {f1_scores[worst_class_idx]:.4f}\n\n")
            
            # Confidence analysis
            f.write("="*80 + "\n")
            f.write("CONFIDENCE ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write(f"Correct Predictions:\n")
            f.write(f"  Mean Confidence: {confidence_stats['correct_mean']:.4f}\n")
            f.write(f"  Median Confidence: {confidence_stats['correct_median']:.4f}\n\n")
            f.write(f"Incorrect Predictions:\n")
            f.write(f"  Mean Confidence: {confidence_stats['incorrect_mean']:.4f}\n")
            f.write(f"  Median Confidence: {confidence_stats['incorrect_median']:.4f}\n\n")
            
            confidence_gap = confidence_stats['correct_mean'] - confidence_stats['incorrect_mean']
            f.write(f"Confidence Gap: {confidence_gap:.4f}\n")
            if confidence_gap > 0.2:
                f.write("✅ Model shows good confidence calibration\n")
            elif confidence_gap > 0.1:
                f.write("⚠️  Model shows moderate confidence calibration\n")
            else:
                f.write("❌ Model may need confidence calibration\n")
            
            # Misclassification analysis
            f.write("\n" + "="*80 + "\n")
            f.write("TOP MISCLASSIFICATION PATTERNS\n")
            f.write("="*80 + "\n")
            
            if len(misclass_df) > 0:
                top_misclass = misclass_df.head(10)
                for idx, row in top_misclass.iterrows():
                    f.write(f"\n{row['True Class']} → {row['Predicted Class']}\n")
                    f.write(f"  Count: {row['Count']}\n")
                    f.write(f"  Percentage: {row['Percentage of True Class']:.1f}%\n")
            else:
                f.write("\nNo misclassifications found (Perfect accuracy!)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"  ✓ Detailed report saved: {report_path.name}")
    
    def evaluate_model(self, model_name, model_path, X_test, y_test, 
                      X_test_original=None, model_type='keras'):
        """
        Complete evaluation of a single model
        
        Args:
            model_name: Name of the model
            model_path: Path to model file
            X_test: Test data (normalized)
            y_test: Test labels
            X_test_original: Original test data for YOLO
            model_type: 'keras' or 'yolo'
            
        Returns:
            Dictionary of evaluation results
        """
        print("\n" + "="*60)
        print(f"EVALUATING: {model_name}")
        print("="*60)
        
        # Load model
        print("Loading model...", end=" ")
        model = self.load_model(model_path, model_type)
        print("✓")
        
        # Generate predictions
        print("Generating predictions...", end=" ")
        if model_type == 'keras':
            y_pred_proba = self.predict_keras(model, X_test)
        else:  # yolo
            y_pred_proba = self.predict_yolo(model, X_test_original)
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        print("✓")
        
        # Calculate metrics
        print("Calculating metrics...", end=" ")
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        print("✓")
        
        # Generate visualizations
        print("Creating visualizations...")
        print("  - Confusion matrices...", end=" ")
        self.plot_confusion_matrix(y_test, y_pred, model_name, normalize=False)
        self.plot_confusion_matrix(y_test, y_pred, model_name, normalize=True)
        print("✓")
        
        print("  - Per-class metrics...", end=" ")
        self.plot_per_class_metrics(metrics, model_name)
        print("✓")
        
        print("  - Misclassification analysis...", end=" ")
        misclass_df = self.analyze_misclassifications(y_test, y_pred, model_name)
        print("✓")
        
        print("  - Confidence distribution...", end=" ")
        confidence_stats = self.plot_confidence_distribution(
            y_pred_proba, y_test, y_pred, model_name
        )
        print("✓")
        
        # Create detailed report
        print("Creating detailed report...", end=" ")
        self.create_detailed_report(
            model_name, metrics, confidence_stats, misclass_df, y_test, y_pred
        )
        print("✓")
        
        # Save metrics to JSON
        metrics_file = self.eval_metrics_path / f'{model_name.replace(" ", "_")}_metrics.json'
        metrics_to_save = {
            'model_name': model_name,
            'overall_metrics': {
                'accuracy': float(metrics['accuracy']),
                'precision_macro': float(metrics['precision_macro']),
                'precision_weighted': float(metrics['precision_weighted']),
                'recall_macro': float(metrics['recall_macro']),
                'recall_weighted': float(metrics['recall_weighted']),
                'f1_macro': float(metrics['f1_macro']),
                'f1_weighted': float(metrics['f1_weighted'])
            },
            'per_class_metrics': metrics['per_class'],
            'confidence_stats': confidence_stats
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        
        print(f"\n✓ Evaluation complete for {model_name}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_macro']:.4f}")
        
        self.evaluation_results[model_name] = metrics_to_save
        
        return metrics_to_save
    
    def compare_all_models(self):
        """
        Create comprehensive comparison of all evaluated models
        """
        if not self.evaluation_results:
            print("\n⚠️  No evaluation results available")
            return
        
        print("\n" + "="*60)
        print("CREATING MODEL COMPARISON")
        print("="*60)
        
        # Extract comparison data
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            metrics = results['overall_metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision (Macro)': metrics['precision_macro'],
                'Recall (Macro)': metrics['recall_macro'],
                'F1-Score (Macro)': metrics['f1_macro'],
                'Precision (Weighted)': metrics['precision_weighted'],
                'Recall (Weighted)': metrics['recall_weighted'],
                'F1-Score (Weighted)': metrics['f1_weighted']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        # Save comparison
        comparison_df.to_csv(self.eval_metrics_path / 'models_comparison.csv', index=False)
        print(f"✓ Comparison saved: models_comparison.csv")
        
        # Display comparison
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        print("\n" + comparison_df.to_string(index=False))
        
        # Plot comparison
        self.plot_models_comparison(comparison_df)
        
        return comparison_df
    
    def plot_models_comparison(self, comparison_df):
        """
        Create visual comparison of all models
        
        Args:
            comparison_df: DataFrame with comparison metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        models = comparison_df['Model'].values
        x_pos = np.arange(len(models))
        
        metrics_to_plot = [
            ('Accuracy', 'skyblue'),
            ('F1-Score (Macro)', 'lightcoral'),
            ('Precision (Macro)', 'lightgreen'),
            ('Recall (Macro)', 'plum')
        ]
        
        for idx, (metric, color) in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            values = comparison_df[metric].values
            
            bars = ax.barh(x_pos, values, color=color, edgecolor='navy', linewidth=1.5)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val + 0.01, i, f'{val:.4f}', 
                       va='center', fontsize=10, fontweight='bold')
            
            ax.set_yticks(x_pos)
            ax.set_yticklabels(models, fontsize=10)
            ax.set_xlabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1.1)
            ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle('All Models Performance Comparison', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.eval_plots_path / 'all_models_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Comparison plot saved: all_models_comparison.png")
    
    def create_final_summary_report(self):
        """
        Create final comprehensive summary report
        """
        if not self.evaluation_results:
            print("\n⚠️  No evaluation results available")
            return
        
        report_path = self.reports_path / 'evaluation_summary.txt'
        
        # Find best models
        best_accuracy = max(r['overall_metrics']['accuracy'] 
                          for r in self.evaluation_results.values())
        best_f1 = max(r['overall_metrics']['f1_macro'] 
                     for r in self.evaluation_results.values())
        
        best_acc_model = [name for name, r in self.evaluation_results.items() 
                         if r['overall_metrics']['accuracy'] == best_accuracy][0]
        best_f1_model = [name for name, r in self.evaluation_results.items() 
                        if r['overall_metrics']['f1_macro'] == best_f1][0]
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("WASTE CLASSIFICATION - EVALUATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("MODELS EVALUATED\n")
            f.write("-"*80 + "\n")
            for model_name in self.evaluation_results.keys():
                f.write(f"  • {model_name}\n")
            
            f.write(f"\nTotal Models: {len(self.evaluation_results)}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("BEST PERFORMING MODELS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"🏆 Best Accuracy: {best_acc_model}\n")
            f.write(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)\n\n")
            
            f.write(f"🏆 Best F1-Score: {best_f1_model}\n")
            f.write(f"   F1-Score: {best_f1:.4f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("DETAILED PERFORMANCE BREAKDOWN\n")
            f.write("="*80 + "\n\n")
            
            for model_name, results in sorted(self.evaluation_results.items(),
                                            key=lambda x: x[1]['overall_metrics']['accuracy'],
                                            reverse=True):
                metrics = results['overall_metrics']
                f.write(f"{model_name}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision_macro']:.4f}\n")
                f.write(f"  Recall: {metrics['recall_macro']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_macro']:.4f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("PER-CLASS ANALYSIS ACROSS ALL MODELS\n")
            f.write("="*80 + "\n\n")
            
            # Calculate average per-class performance
            for class_idx, class_name in enumerate(self.class_names):
                f.write(f"{class_name}:\n")
                
                f1_scores = [r['per_class_metrics']['f1'][class_idx] 
                            for r in self.evaluation_results.values()]
                
                f.write(f"  Average F1-Score: {np.mean(f1_scores):.4f}\n")
                f.write(f"  Best F1-Score: {np.max(f1_scores):.4f}\n")
                f.write(f"  Worst F1-Score: {np.min(f1_scores):.4f}\n")
                f.write(f"  Std Dev: {np.std(f1_scores):.4f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")
            
            f.write("For Deployment:\n")
            f.write(f"  Primary Model: {best_acc_model}\n")
            f.write(f"  Reason: Highest accuracy ({best_accuracy:.4f})\n\n")
            
            if best_f1_model != best_acc_model:
                f.write(f"  Alternative Model: {best_f1_model}\n")
                f.write(f"  Reason: Best balanced performance (F1: {best_f1:.4f})\n\n")
            
            # Identify challenging classes
            avg_f1_per_class = {}
            for class_idx, class_name in enumerate(self.class_names):
                f1_scores = [r['per_class_metrics']['f1'][class_idx] 
                            for r in self.evaluation_results.values()]
                avg_f1_per_class[class_name] = np.mean(f1_scores)
            
            sorted_classes = sorted(avg_f1_per_class.items(), key=lambda x: x[1])
            
            f.write("Most Challenging Classes (need improvement):\n")
            for class_name, avg_f1 in sorted_classes[:3]:
                f.write(f"  • {class_name} (Avg F1: {avg_f1:.4f})\n")
            
            f.write("\nBest Performing Classes:\n")
            for class_name, avg_f1 in sorted_classes[-3:]:
                f.write(f"  • {class_name} (Avg F1: {avg_f1:.4f})\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF EVALUATION SUMMARY\n")
            f.write("="*80 + "\n")
        
        print(f"\n✓ Final summary report saved: {report_path.name}")


def main():
    """Main execution function"""
    print("="*60)
    print("WASTE CLASSIFICATION - MODEL EVALUATION")
    print("="*60)
    
    # Check if data and models exist
    results_path = Path('results')
    processed_data_path = results_path / 'processed_data'
    models_path = results_path / 'models'
    
    if not processed_data_path.exists():
        print("\n❌ Error: Processed data not found!")
        print("Please run data_preparation.py first.")
        return
    
    if not models_path.exists():
        print("\n❌ Error: Models directory not found!")
        print("Please run train_models.py first.")
        return
    
    # Load test data
    print("\nLoading test data...")
    X_test = np.load(processed_data_path / 'X_test.npy')
    y_test = np.load(processed_data_path / 'y_test.npy')
    
    print(f"✓ Test data: {X_test.shape}")
    print(f"✓ Test labels: {y_test.shape}")
    
    # Normalize for Keras models
    X_test_normalized = X_test / 255.0
    
    # Load data info
    with open(results_path / 'metrics' / 'data_split_info.json', 'r') as f:
        data_info = json.load(f)
    
    class_names = data_info['class_names']
    num_classes = data_info['num_classes']
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    # Initialize evaluator
    evaluator = WasteModelEvaluator(
        results_path='results',
        num_classes=num_classes,
        class_names=class_names
    )
    
    # Find available models
    print("\n" + "="*60)
    print("AVAILABLE MODELS")
    print("="*60)
    
    keras_models = list(models_path.glob('*_best.h5'))
    yolo_models = list(models_path.glob('YOLO*_best.pt'))
    
    print(f"\nKeras models: {len(keras_models)}")
    for model_path in keras_models:
        model_name = model_path.stem.replace('_best', '')
        print(f"  • {model_name}")
    
    if yolo_models and YOLO_AVAILABLE:
        print(f"\nYOLO models: {len(yolo_models)}")
        for model_path in yolo_models:
            model_name = model_path.stem.replace('_best', '')
            print(f"  • {model_name}")
    
    total_models = len(keras_models) + (len(yolo_models) if YOLO_AVAILABLE else 0)
    
    if total_models == 0:
        print("\n❌ Error: No trained models found!")
        print("Please run train_models.py first.")
        return
    
    # Model selection
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)
    print("\nOptions:")
    print("  1. Evaluate all models (recommended)")
    print("  2. Evaluate only Keras models")
    print("  3. Evaluate only YOLO models")
    print("  4. Evaluate specific models")
    print("  5. Evaluate best model only")
    
    choice = input("\nSelect option (1-5) [default: 1]: ").strip() or "1"
    
    models_to_evaluate = []
    
    if choice == "1":
        # All models
        for model_path in keras_models:
            model_name = model_path.stem.replace('_best', '')
            models_to_evaluate.append((model_name, model_path, 'keras'))
        
        if YOLO_AVAILABLE:
            for model_path in yolo_models:
                model_name = model_path.stem.replace('_best', '')
                models_to_evaluate.append((model_name, model_path, 'yolo'))
    
    elif choice == "2":
        # Only Keras
        for model_path in keras_models:
            model_name = model_path.stem.replace('_best', '')
            models_to_evaluate.append((model_name, model_path, 'keras'))
    
    elif choice == "3":
        # Only YOLO
        if YOLO_AVAILABLE:
            for model_path in yolo_models:
                model_name = model_path.stem.replace('_best', '')
                models_to_evaluate.append((model_name, model_path, 'yolo'))
        else:
            print("\n⚠️  YOLO not available")
            return
    
    elif choice == "4":
        # Custom selection
        print("\nAvailable models:")
        all_models = []
        for model_path in keras_models:
            model_name = model_path.stem.replace('_best', '')
            all_models.append((model_name, model_path, 'keras'))
            print(f"  • {model_name}")
        
        if YOLO_AVAILABLE:
            for model_path in yolo_models:
                model_name = model_path.stem.replace('_best', '')
                all_models.append((model_name, model_path, 'yolo'))
                print(f"  • {model_name}")
        
        print("\nEnter model names separated by commas:")
        selected = input("Models: ").strip().split(',')
        selected = [s.strip() for s in selected]
        
        for model_name, model_path, model_type in all_models:
            if model_name in selected:
                models_to_evaluate.append((model_name, model_path, model_type))
    
    else:
        # Best model only
        summary_file = results_path / 'metrics' / 'training_summary.csv'
        if summary_file.exists():
            summary_df = pd.read_csv(summary_file)
            best_model_name = summary_df.iloc[0]['Model']
            
            # Find the model file
            for model_path in keras_models:
                if model_path.stem.replace('_best', '') == best_model_name:
                    models_to_evaluate.append((best_model_name, model_path, 'keras'))
                    break
            
            if YOLO_AVAILABLE:
                for model_path in yolo_models:
                    if model_path.stem.replace('_best', '') == best_model_name:
                        models_to_evaluate.append((best_model_name, model_path, 'yolo'))
                        break
            
            if not models_to_evaluate:
                print(f"\n⚠️  Best model '{best_model_name}' not found")
                return
        else:
            print("\n⚠️  Training summary not found, evaluating all models")
            choice = "1"
    
    if not models_to_evaluate:
        print("\n⚠️  No models selected for evaluation")
        return
    
    print(f"\n✓ Selected {len(models_to_evaluate)} model(s) for evaluation")
    
    # Confirm evaluation
    print("\n" + "="*60)
    print("EVALUATION PLAN")
    print("="*60)
    print(f"\nModels to evaluate: {len(models_to_evaluate)}")
    for model_name, _, model_type in models_to_evaluate:
        print(f"  • {model_name} ({model_type})")
    
    print("\nThis will generate:")
    print("  • Confusion matrices (normalized and counts)")
    print("  • Per-class performance metrics")
    print("  • Misclassification analysis")
    print("  • Confidence distribution analysis")
    print("  • Detailed evaluation reports")
    print("  • Comprehensive comparison")
    
    response = input("\nProceed with evaluation? (yes/no) [yes]: ").strip().lower()
    if response and response not in ['yes', 'y']:
        print("Evaluation cancelled.")
        return
    
    # Evaluate each model
    print("\n" + "="*60)
    print("STARTING MODEL EVALUATION")
    print("="*60)
    
    start_time = time.time()
    
    for model_name, model_path, model_type in models_to_evaluate:
        try:
            evaluator.evaluate_model(
                model_name=model_name,
                model_path=model_path,
                X_test=X_test_normalized,
                y_test=y_test,
                X_test_original=X_test,
                model_type=model_type
            )
        except Exception as e:
            print(f"\n❌ Error evaluating {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - start_time
    
    # Create comparison and summary
    print("\n" + "="*60)
    print("CREATING FINAL COMPARISON AND SUMMARY")
    print("="*60)
    
    comparison_df = evaluator.compare_all_models()
    evaluator.create_final_summary_report()
    
    # Final summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    
    print(f"\n✓ Total time: {total_time/60:.2f} minutes")
    print(f"✓ Models evaluated: {len(evaluator.evaluation_results)}")
    
    if comparison_df is not None and len(comparison_df) > 0:
        best_model = comparison_df.iloc[0]
        print(f"\n🏆 Best Model: {best_model['Model']}")
        print(f"   Accuracy: {best_model['Accuracy']:.4f}")
        print(f"   F1-Score: {best_model['F1-Score (Macro)']:.4f}")
    
    print("\n" + "="*60)
    print("GENERATED FILES")
    print("="*60)
    
    print("\n📊 Metrics:")
    print("  • evaluation/models_comparison.csv")
    print("  • evaluation/*_metrics.json")
    print("  • evaluation/*_misclassification_report.csv")
    
    print("\n📈 Plots:")
    print("  • evaluation/*_confusion_matrix_*.png")
    print("  • evaluation/*_per_class_metrics.png")
    print("  • evaluation/*_misclassification_matrix.png")
    print("  • evaluation/*_confidence_distribution.png")
    print("  • evaluation/all_models_comparison.png")
    
    print("\n📄 Reports:")
    print("  • reports/*_detailed_report.txt")
    print("  • reports/evaluation_summary.txt")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Review evaluation reports in results/reports/")
    print("2. Check confusion matrices for misclassification patterns")
    print("3. Analyze per-class performance for weak classes")
    print("4. Run predict.py to test on new images")
    print("5. Deploy best model for production use")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()