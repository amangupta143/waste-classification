"""
Standalone YOLOv8 Training Script for Waste Classification
Train YOLOv8n and YOLOv8s models independently
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import shutil
from tqdm import tqdm

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ Error: ultralytics not installed.")
    print("Install with: pip install ultralytics")
    exit(1)


class YOLOWasteTrainer:
    """Standalone YOLO trainer for waste classification"""
    
    def __init__(self, results_path='results'):
        self.results_path = Path(results_path)
        self.models_path = self.results_path / 'models'
        self.plots_path = self.results_path / 'plots'
        self.metrics_path = self.results_path / 'metrics'
        self.yolo_data_path = self.results_path / 'yolo_split_dataset'
        
        # Create directories
        for path in [self.models_path, self.plots_path, self.metrics_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Check if YOLO dataset exists
        if not self.yolo_data_path.exists():
            print(f"❌ Error: YOLO dataset not found at {self.yolo_data_path}")
            print("Please run data_preparation.py first to create the dataset.")
            exit(1)
        
        # Load dataset info
        info_file = self.results_path / 'metrics' / 'data_split_info.json'
        if info_file.exists():
            with open(info_file, 'r') as f:
                data_info = json.load(f)
            self.class_names = data_info['class_names']
            self.num_classes = data_info['num_classes']
        else:
            # Infer from dataset
            self.class_names = sorted([d.name for d in (self.yolo_data_path / 'train').iterdir() if d.is_dir()])
            self.num_classes = len(self.class_names)
        
        print(f"✓ YOLO Trainer initialized")
        print(f"✓ Classes: {self.num_classes}")
        print(f"✓ Dataset: {self.yolo_data_path}")
        
        self.training_histories = {}
    
    def train_yolo_model(self, model_size='n', epochs=100, imgsz=640, batch=8):
        """
        Train YOLOv8 classification model
        
        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size (keep lower for memory)
        """
        model_name = f'YOLOv8{model_size}'
        
        print("\n" + "="*70)
        print(f"TRAINING {model_name.upper()} CLASSIFICATION MODEL")
        print("="*70)
        
        # Load pretrained model
        yolo_model_path = f'yolov8{model_size}-cls.pt'
        print(f"\nLoading pretrained model: {yolo_model_path}")
        model = YOLO(yolo_model_path)
        
        # Training configuration
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Image size: {imgsz}")
        print(f"  Batch size: {batch}")
        print(f"  Dataset: {self.yolo_data_path}")
        print(f"  Optimizer: AdamW")
        print(f"  Learning rate: 0.001 → 0.0001 (cosine)")
        print(f"  Patience: 15")
        
        # Record start time
        start_time = time.time()
        
        # Train model
        print(f"\n🚀 Starting training...")
        results = model.train(
            data=str(self.yolo_data_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=15,
            lr0=0.001,
            lrf=0.0001,
            weight_decay=0.0005,
            optimizer='AdamW',
            device=0,  # Use GPU
            augment=False,  # Already augmented
            verbose=True,
            workers=2,
            cos_lr=True,  # Cosine annealing
            project=str(self.models_path / 'yolo_runs'),
            name=model_name,
            exist_ok=True,
            plots=True,
            save=True,
            save_period=-1
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {training_time/60:.2f} minutes")
        
        # Validate on test set
        print(f"\n📊 Validating on test set...")
        metrics = model.val(
            data=str(self.yolo_data_path / 'test'),
            imgsz=imgsz,
            batch=batch,
            verbose=True
        )
        
        # Extract metrics
        top1_acc = float(metrics.top1) if hasattr(metrics, 'top1') else 0.0
        top5_acc = float(metrics.top5) if hasattr(metrics, 'top5') else 0.0
        
        print(f"\n" + "="*70)
        print(f"{model_name} RESULTS")
        print("="*70)
        print(f"  Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
        print(f"  Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
        print(f"  Training Time: {training_time/60:.2f} minutes")
        print(f"  Epochs Trained: {epochs}")
        print("="*70)
        
        # Save best model
        best_model_src = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
        best_model_dst = self.models_path / f'{model_name}_best.pt'
        if best_model_src.exists():
            shutil.copy(best_model_src, best_model_dst)
            print(f"\n✓ Best model saved to: {best_model_dst}")
        
        # Save last model
        last_model_src = Path(model.trainer.save_dir) / 'weights' / 'last.pt'
        last_model_dst = self.models_path / f'{model_name}_last.pt'
        if last_model_src.exists():
            shutil.copy(last_model_src, last_model_dst)
            print(f"✓ Last model saved to: {last_model_dst}")
        
        # Plot training results
        results_csv = Path(model.trainer.save_dir) / 'results.csv'
        if results_csv.exists():
            print(f"\n📈 Creating training plots...")
            self.plot_yolo_history(model_name, results_csv)
        
        # Store history
        self.training_histories[model_name] = {
            'training_time': training_time,
            'epochs_trained': epochs,
            'top1_accuracy': top1_acc,
            'top5_accuracy': top5_acc,
            'model_size': model_size,
            'parameters': self.get_model_params(model_size)
        }
        
        # Save history to JSON
        history_data = {
            'model_name': model_name,
            'model_size': model_size,
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'epochs_trained': epochs,
            'top1_accuracy': top1_acc,
            'top5_accuracy': top5_acc,
            'imgsz': imgsz,
            'batch_size': batch,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.0001,
            'best_model_path': str(best_model_dst)
        }
        
        history_path = self.metrics_path / f'{model_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=4)
        
        print(f"✓ Training history saved to: {history_path}")
        
        return results, metrics
    
    def get_model_params(self, model_size):
        """Get approximate parameter count for YOLO model"""
        params_map = {
            'n': '1.45M',
            's': '5.09M',
            'm': '12.2M',
            'l': '23.4M',
            'x': '45.3M'
        }
        return params_map.get(model_size, 'Unknown')
    
    def plot_yolo_history(self, model_name, results_csv):
        """Plot YOLO training history from results CSV"""
        df = pd.read_csv(results_csv)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Loss plot
        if 'train/loss' in df.columns and 'val/loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['train/loss'], 
                           label='Train Loss', linewidth=2, color='#2E86AB')
            axes[0, 0].plot(df['epoch'], df['val/loss'], 
                           label='Val Loss', linewidth=2, color='#A23B72')
            axes[0, 0].set_title(f'{model_name} - Loss Curve', 
                               fontsize=14, fontweight='bold', pad=15)
            axes[0, 0].set_xlabel('Epoch', fontsize=12)
            axes[0, 0].set_ylabel('Loss', fontsize=12)
            axes[0, 0].legend(fontsize=11)
            axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        
        # Accuracy plot
        if 'metrics/accuracy_top1' in df.columns:
            axes[0, 1].plot(df['epoch'], df['metrics/accuracy_top1'], 
                           label='Top-1 Accuracy', linewidth=2.5, color='#06A77D')
            if 'metrics/accuracy_top5' in df.columns:
                axes[0, 1].plot(df['epoch'], df['metrics/accuracy_top5'], 
                               label='Top-5 Accuracy', linewidth=2, 
                               color='#F18F01', linestyle='--')
            axes[0, 1].set_title(f'{model_name} - Accuracy Progress', 
                               fontsize=14, fontweight='bold', pad=15)
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('Accuracy', fontsize=12)
            axes[0, 1].legend(fontsize=11)
            axes[0, 1].grid(True, alpha=0.3, linestyle='--')
            axes[0, 1].set_ylim([0, 1.05])
        
        # Learning rate plot
        if 'lr/pg0' in df.columns:
            axes[1, 0].plot(df['epoch'], df['lr/pg0'], 
                           linewidth=2, color='#D62828')
            axes[1, 0].set_title(f'{model_name} - Learning Rate Schedule', 
                               fontsize=14, fontweight='bold', pad=15)
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
            axes[1, 0].grid(True, alpha=0.3, linestyle='--')
            axes[1, 0].set_yscale('log')
        
        # Best metrics summary
        if 'metrics/accuracy_top1' in df.columns:
            best_acc = df['metrics/accuracy_top1'].max()
            best_epoch = df.loc[df['metrics/accuracy_top1'].idxmax(), 'epoch']
            final_acc = df['metrics/accuracy_top1'].iloc[-1]
            
            summary_text = f"Training Summary:\n\n"
            summary_text += f"Best Top-1 Accuracy: {best_acc:.4f}\n"
            summary_text += f"Best Epoch: {int(best_epoch)}\n"
            summary_text += f"Final Accuracy: {final_acc:.4f}\n"
            summary_text += f"Total Epochs: {len(df)}\n"
            
            if 'train/loss' in df.columns:
                final_loss = df['train/loss'].iloc[-1]
                summary_text += f"Final Train Loss: {final_loss:.4f}\n"
            
            axes[1, 1].text(0.1, 0.5, summary_text, 
                          transform=axes[1, 1].transAxes,
                          fontsize=13, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='wheat', 
                                  alpha=0.3, pad=1))
            axes[1, 1].axis('off')
            axes[1, 1].set_title(f'{model_name} - Summary', 
                               fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        plot_path = self.plots_path / f'{model_name}_training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training plots saved to: {plot_path}")
    
    def create_comparison_summary(self):
        """Create summary comparing all trained YOLO models"""
        if not self.training_histories:
            print("No training histories available")
            return
        
        print("\n" + "="*70)
        print("CREATING YOLO MODELS COMPARISON")
        print("="*70)
        
        summary_data = []
        for model_name, data in self.training_histories.items():
            summary_data.append({
                'Model': model_name,
                'Parameters': data['parameters'],
                'Top-1 Accuracy': data['top1_accuracy'],
                'Top-5 Accuracy': data['top5_accuracy'],
                'Training Time (min)': data['training_time'] / 60,
                'Epochs': data['epochs_trained']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Top-1 Accuracy', ascending=False)
        
        # Save to CSV
        summary_csv = self.metrics_path / 'yolo_models_summary.csv'
        summary_df.to_csv(summary_csv, index=False)
        
        print(f"\n✓ Summary saved to: {summary_csv}")
        
        print("\n" + "="*70)
        print("YOLO MODELS COMPARISON")
        print("="*70)
        print(summary_df.to_string(index=False))
        print("="*70)
        
        # Plot comparison
        if len(summary_data) > 1:
            self.plot_yolo_comparison(summary_df)
        
        return summary_df
    
    def plot_yolo_comparison(self, summary_df):
        """Plot comparison between YOLO models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = summary_df['Model'].values
        colors = ['#2E86AB', '#A23B72', '#06A77D', '#F18F01']
        
        # Top-1 Accuracy
        axes[0, 0].barh(models, summary_df['Top-1 Accuracy'], 
                       color=colors[:len(models)], edgecolor='black', linewidth=1.5)
        axes[0, 0].set_xlabel('Accuracy', fontsize=12)
        axes[0, 0].set_title('Top-1 Accuracy Comparison', 
                           fontsize=13, fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)
        for i, v in enumerate(summary_df['Top-1 Accuracy']):
            axes[0, 0].text(v + 0.01, i, f'{v:.4f}', 
                          va='center', fontsize=10, fontweight='bold')
        
        # Training Time
        axes[0, 1].barh(models, summary_df['Training Time (min)'], 
                       color=colors[:len(models)], edgecolor='black', linewidth=1.5)
        axes[0, 1].set_xlabel('Time (minutes)', fontsize=12)
        axes[0, 1].set_title('Training Time Comparison', 
                           fontsize=13, fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        for i, v in enumerate(summary_df['Training Time (min)']):
            axes[0, 1].text(v + 1, i, f'{v:.1f}', 
                          va='center', fontsize=10, fontweight='bold')
        
        # Parameters
        axes[1, 0].barh(models, range(len(models)), 
                       color=colors[:len(models)], edgecolor='black', linewidth=1.5)
        axes[1, 0].set_yticks(range(len(models)))
        axes[1, 0].set_yticklabels(models)
        axes[1, 0].set_xlabel('Model Complexity', fontsize=12)
        axes[1, 0].set_title('Model Parameters', 
                           fontsize=13, fontweight='bold')
        for i, (model, params) in enumerate(zip(models, summary_df['Parameters'])):
            axes[1, 0].text(0.5, i, params, 
                          va='center', ha='center', 
                          fontsize=11, fontweight='bold', color='white')
        axes[1, 0].set_xticks([])
        
        # Accuracy vs Time scatter
        axes[1, 1].scatter(summary_df['Training Time (min)'], 
                          summary_df['Top-1 Accuracy'],
                          s=300, c=colors[:len(models)], 
                          edgecolors='black', linewidth=2, alpha=0.7)
        for i, model in enumerate(models):
            axes[1, 1].annotate(model, 
                               (summary_df['Training Time (min)'].iloc[i],
                                summary_df['Top-1 Accuracy'].iloc[i]),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', 
                                       facecolor='yellow', alpha=0.3))
        axes[1, 1].set_xlabel('Training Time (minutes)', fontsize=12)
        axes[1, 1].set_ylabel('Top-1 Accuracy', fontsize=12)
        axes[1, 1].set_title('Accuracy vs Training Time', 
                           fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        comparison_path = self.plots_path / 'yolo_models_comparison.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison plot saved to: {comparison_path}")
    
    def update_overall_summary(self):
        """Update the overall training summary to include YOLO models"""
        overall_summary = self.metrics_path / 'training_summary.csv'
        
        if not overall_summary.exists():
            print("\n⚠️  Overall training summary not found. Creating YOLO-only summary.")
            return
        
        # Load existing summary
        existing_df = pd.read_csv(overall_summary)
        
        # Add YOLO models
        yolo_data = []
        for model_name, data in self.training_histories.items():
            yolo_data.append({
                'Model': model_name,
                'Training Time (min)': data['training_time'] / 60,
                'Epochs Trained': data['epochs_trained'],
                'Final Train Accuracy': data['top1_accuracy'],  # YOLO doesn't separate
                'Final Val Accuracy': data['top1_accuracy'],
                'Best Val Accuracy': data['top1_accuracy'],
                'Final Train Loss': 0.0,  # Not tracked separately
                'Final Val Loss': 0.0,
                'Best Val Loss': 0.0
            })
        
        yolo_df = pd.DataFrame(yolo_data)
        
        # Combine
        combined_df = pd.concat([existing_df, yolo_df], ignore_index=True)
        combined_df = combined_df.sort_values('Best Val Accuracy', ascending=False)
        
        # Save updated summary
        combined_df.to_csv(overall_summary, index=False)
        
        print(f"\n✓ Updated overall summary: {overall_summary}")
        print("\n" + "="*70)
        print("UPDATED OVERALL SUMMARY - ALL MODELS")
        print("="*70)
        print(combined_df.to_string(index=False))
        print("="*70)


def main():
    """Main execution function"""
    print("="*70)
    print("YOLOV8 STANDALONE TRAINING FOR WASTE CLASSIFICATION")
    print("="*70)
    
    # Initialize trainer
    trainer = YOLOWasteTrainer(results_path='results')
    
    # Configuration
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    
    models_to_train = ['n', 's']  # YOLOv8n and YOLOv8s
    epochs = 100
    imgsz = 640
    batch_size = 8  # Reduced for memory safety
    
    print(f"\nModels to train: {['YOLOv8' + m for m in models_to_train]}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch_size}")
    
    # Ask for confirmation
    print("\n" + "="*70)
    response = input("Start YOLO training? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Training cancelled.")
        return
    
    # Train each model
    for model_size in models_to_train:
        try:
            print(f"\n\n{'='*70}")
            print(f"STARTING YOLOv8{model_size.upper()} TRAINING")
            print(f"{'='*70}")
            
            trainer.train_yolo_model(
                model_size=model_size,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size
            )
            
            print(f"\n✓ YOLOv8{model_size} training completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error training YOLOv8{model_size}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison summary
    if trainer.training_histories:
        print("\n\n" + "="*70)
        print("CREATING FINAL SUMMARY")
        print("="*70)
        
        trainer.create_comparison_summary()
        trainer.update_overall_summary()
        
        print("\n" + "="*70)
        print("YOLO TRAINING COMPLETE!")
        print("="*70)
        print(f"\n✓ Trained {len(trainer.training_histories)} YOLO models")
        print(f"✓ All results saved to: results/")
        print(f"✓ Models saved to: results/models/")
        print(f"✓ Plots saved to: results/plots/")
        print(f"✓ Metrics saved to: results/metrics/")
        
        # Show best model
        best_model = max(trainer.training_histories.items(), 
                        key=lambda x: x[1]['top1_accuracy'])
        print(f"\n🏆 Best YOLO Model: {best_model[0]}")
        print(f"   Top-1 Accuracy: {best_model[1]['top1_accuracy']:.4f}")
        print(f"   Training Time: {best_model[1]['training_time']/60:.2f} min")
        
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print("  1. Run evaluate.py to test all models on test set")
        print("  2. Run ensemble.py to create ensemble models")
        print("  3. Run predict.py to make predictions on new images")
    else:
        print("\n❌ No models were trained successfully.")


if __name__ == "__main__":
    main()