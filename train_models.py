import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import (
    ResNet50, VGG16, MobileNetV2, EfficientNetB0, InceptionV3
)
from tensorflow.keras.regularizers import l2


class WasteModelTrainer:
    """
    Handles training of multiple deep learning models for waste classification
    """
    
    def __init__(self, results_path='results', img_shape=(224, 224, 3), 
                 num_classes=9, class_names=None):
        """
        Initialize model trainer
        
        Args:
            results_path: Path to store results
            img_shape: Input image shape (height, width, channels)
            num_classes: Number of classification categories
            class_names: List of class names
        """
        self.results_path = Path(results_path)
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        
        # Create directories
        self.models_path = self.results_path / 'models'
        self.plots_path = self.results_path / 'plots'
        self.metrics_path = self.results_path / 'metrics'
        
        for path in [self.models_path, self.plots_path, self.metrics_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Storage for training histories
        self.training_histories = {}
        
        print(f"Model Trainer initialized")
        print(f"Image shape: {self.img_shape}")
        print(f"Number of classes: {self.num_classes}")
    
    def create_custom_cnn(self, name='CustomCNN'):
        """
        Create an improved custom CNN architecture with BatchNormalization
        Inspired by best practices from successful models
        
        Args:
            name: Model name
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential(name=name)
        
        # Block 1: 32 filters
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                                input_shape=self.img_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Block 2: 64 filters
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Block 3: 128 filters
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Block 4: 256 filters
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Dense layers with strong regularization
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def create_deep_cnn(self, name='DeepCNN'):
        """
        Create a deeper CNN with more layers and BatchNormalization
        Inspired by inspiration_code2
        
        Args:
            name: Model name
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential(name=name)
        
        # Block 1
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                input_shape=self.img_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Block 2
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Block 3
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Block 4
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def create_transfer_learning_model(self, base_model_name='MobileNetV2', trainable_layers=0):
        """
        Create a transfer learning model using pre-trained architectures
        
        Args:
            base_model_name: Name of the base architecture
            trainable_layers: Number of top layers to make trainable (0 = freeze all)
            
        Returns:
            Compiled Keras model
        """
        # Select base model
        base_models = {
            'ResNet50': ResNet50,
            'VGG16': VGG16,
            'MobileNetV2': MobileNetV2,
            'EfficientNetB0': EfficientNetB0,
            'InceptionV3': InceptionV3
        }
        
        if base_model_name not in base_models:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        # Load base model without top layers
        base_model = base_models[base_model_name](
            include_top=False,
            weights='imagenet',
            input_shape=self.img_shape
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Create new model with improved architecture
        inputs = keras.Input(shape=self.img_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs, name=base_model_name)
        
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """
        Compile model with optimizer, loss, and metrics
        
        Args:
            model: Keras model to compile
            learning_rate: Learning rate for optimizer
        """
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        return model
    
    def get_callbacks(self, model_name, patience=5):
        """
        Create callbacks for training with more aggressive early stopping
        
        Args:
            model_name: Name of the model (for file naming)
            patience: Patience for early stopping
            
        Returns:
            List of callbacks
        """
        # Model checkpoint
        checkpoint_path = self.models_path / f'{model_name}_best.h5'
        checkpoint = callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Early stopping with shorter patience
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        
        # CSV logger
        csv_path = self.metrics_path / f'{model_name}_training_log.csv'
        csv_logger = callbacks.CSVLogger(str(csv_path))
        
        return [checkpoint, early_stop, reduce_lr, csv_logger]
    
    def train_model(self, model, model_name, train_generator, val_data, 
                   epochs=50, steps_per_epoch=None):
        """
        Train a single model using data generator
        
        Args:
            model: Keras model to train
            model_name: Name of the model
            train_generator: Training data generator with augmentation
            val_data: Validation data tuple (X_val, y_val)
            epochs: Number of training epochs
            steps_per_epoch: Steps per epoch (if None, uses generator length)
            
        Returns:
            Training history
        """
        print("\n" + "="*60)
        print(f"Training {model_name}")
        print("="*60)
        
        # Model summary
        print(f"\nModel architecture:")
        model.summary()
        
        # Get callbacks
        callback_list = self.get_callbacks(model_name, patience=5)
        
        # Record start time
        start_time = time.time()
        
        # Train with generator
        print(f"\nTraining with data augmentation...")
        history = model.fit(
            train_generator,
            validation_data=val_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callback_list,
            verbose=1
        )
        
        # Record end time
        training_time = time.time() - start_time
        
        # Save final model
        final_model_path = self.models_path / f'{model_name}_final.h5'
        model.save(str(final_model_path))
        
        # Store history
        self.training_histories[model_name] = {
            'history': history.history,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss'])
        }
        
        # Save history to JSON
        history_data = {
            'model_name': model_name,
            'training_time_seconds': training_time,
            'epochs_trained': len(history.history['loss']),
            'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
        }
        
        history_path = self.metrics_path / f'{model_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=4)
        
        print(f"\n{model_name} training completed!")
        print(f"Training time: {training_time/60:.2f} minutes")
        print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
        
        # Plot training history
        self.plot_training_history(model_name, history.history)
        
        return history
    
    def plot_training_history(self, model_name, history):
        """
        Plot training and validation metrics
        
        Args:
            model_name: Name of the model
            history: Training history dictionary
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        axes[0, 0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 0].set_title(f'{model_name} - Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 1].set_title(f'{model_name} - Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Train Precision', linewidth=2)
            axes[1, 0].plot(history['val_precision'], label='Val Precision', linewidth=2)
            axes[1, 0].set_title(f'{model_name} - Precision', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Train Recall', linewidth=2)
            axes[1, 1].plot(history['val_recall'], label='Val Recall', linewidth=2)
            axes[1, 1].set_title(f'{model_name} - Recall', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / f'{model_name}_training_history.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved: {self.plots_path / f'{model_name}_training_history.png'}")
    
    def create_training_summary(self):
        """
        Create a summary comparison of all trained models
        """
        if not self.training_histories:
            print("No training histories available for summary")
            return
        
        print("\n" + "="*60)
        print("CREATING TRAINING SUMMARY")
        print("="*60)
        
        # Collect summary data
        summary_data = []
        for model_name, data in self.training_histories.items():
            history = data['history']
            summary_data.append({
                'Model': model_name,
                'Training Time (min)': data['training_time'] / 60,
                'Epochs Trained': data['epochs_trained'],
                'Final Train Accuracy': history['accuracy'][-1],
                'Final Val Accuracy': history['val_accuracy'][-1],
                'Best Val Accuracy': max(history['val_accuracy']),
                'Final Train Loss': history['loss'][-1],
                'Final Val Loss': history['val_loss'][-1],
                'Best Val Loss': min(history['val_loss'])
            })
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Best Val Accuracy', ascending=False)
        
        # Save to CSV
        summary_df.to_csv(self.metrics_path / 'training_summary.csv', index=False)
        print(f"\nTraining summary saved to: {self.metrics_path / 'training_summary.csv'}")
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY - ALL MODELS")
        print("="*60)
        print(summary_df.to_string(index=False))
        
        # Create comparison plots
        self.plot_model_comparison(summary_df)
        
        return summary_df
    
    def plot_model_comparison(self, summary_df):
        """
        Create comparison plots for all models
        
        Args:
            summary_df: DataFrame with model summaries
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models = summary_df['Model'].values
        
        # Best Validation Accuracy
        axes[0, 0].barh(models, summary_df['Best Val Accuracy'], color='skyblue', edgecolor='navy')
        axes[0, 0].set_xlabel('Accuracy', fontsize=11)
        axes[0, 0].set_title('Best Validation Accuracy', fontsize=13, fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Training Time
        axes[0, 1].barh(models, summary_df['Training Time (min)'], color='lightcoral', edgecolor='darkred')
        axes[0, 1].set_xlabel('Time (minutes)', fontsize=11)
        axes[0, 1].set_title('Training Time', fontsize=13, fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # Final Val Loss
        axes[1, 0].barh(models, summary_df['Final Val Loss'], color='lightgreen', edgecolor='darkgreen')
        axes[1, 0].set_xlabel('Loss', fontsize=11)
        axes[1, 0].set_title('Final Validation Loss', fontsize=13, fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Epochs Trained
        axes[1, 1].barh(models, summary_df['Epochs Trained'], color='plum', edgecolor='purple')
        axes[1, 1].set_xlabel('Epochs', fontsize=11)
        axes[1, 1].set_title('Epochs Trained', fontsize=13, fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'models_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model comparison plot saved: {self.plots_path / 'models_comparison.png'}")
        
        # Create accuracy comparison over epochs
        self.plot_accuracy_comparison()
    
    def plot_accuracy_comparison(self):
        """
        Plot validation accuracy comparison across all models
        """
        plt.figure(figsize=(14, 8))
        
        for model_name, data in self.training_histories.items():
            history = data['history']
            epochs = range(1, len(history['val_accuracy']) + 1)
            plt.plot(epochs, history['val_accuracy'], label=model_name, linewidth=2, marker='o', markersize=4)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation Accuracy', fontsize=12)
        plt.title('Validation Accuracy Comparison - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_path / 'accuracy_comparison_all_models.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Accuracy comparison plot saved: {self.plots_path / 'accuracy_comparison_all_models.png'}")
    
    def train_all_models(self, X_train, y_train, X_val, y_val, 
                        models_to_train=None, epochs=50, batch_size=32, 
                        train_datagen=None, val_datagen=None, learning_rate=0.001):
        """
        Train multiple models using data generators
        
        Args:
            X_train, y_train: Training data (UNNORMALIZED)
            X_val, y_val: Validation data (UNNORMALIZED)
            models_to_train: List of model names to train
            epochs: Number of epochs
            batch_size: Batch size
            train_datagen: Training data augmentation generator (with rescale)
            val_datagen: Validation data generator (with rescale only)
            learning_rate: Learning rate
            
        Returns:
            Dictionary of trained models
        """
        if models_to_train is None:
            models_to_train = ['CustomCNN', 'DeepCNN', 'MobileNetV2', 'ResNet50', 'VGG16']
        
        print("\n" + "="*60)
        print("WASTE CLASSIFICATION - MODEL TRAINING")
        print("="*60)
        print(f"\nModels to train: {models_to_train}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        
        # Create generators if not provided
        if train_datagen is None or val_datagen is None:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                width_shift_range=0.25,
                height_shift_range=0.25,
                shear_range=0.25,
                zoom_range=0.25,
                horizontal_flip=True,
                fill_mode='nearest',
                brightness_range=[0.7, 1.3]
            )
            val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create training generator
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Normalize validation data
        X_val_normalized = val_datagen.flow(
            X_val, y_val,
            batch_size=len(X_val),
            shuffle=False
        ).next()
        
        # Calculate steps per epoch
        steps_per_epoch = len(X_train) // batch_size
        
        trained_models = {}
        
        for model_name in models_to_train:
            try:
                # Create model
                if model_name == 'CustomCNN':
                    model = self.create_custom_cnn(name=model_name)
                elif model_name == 'DeepCNN':
                    model = self.create_deep_cnn(name=model_name)
                else:
                    model = self.create_transfer_learning_model(
                        base_model_name=model_name,
                        trainable_layers=0
                    )
                
                # Compile model
                model = self.compile_model(model, learning_rate=learning_rate)
                
                # Train model
                history = self.train_model(
                    model, model_name,
                    train_generator, X_val_normalized,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch
                )
                
                trained_models[model_name] = model
                
            except Exception as e:
                print(f"\nError training {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create comparison summary
        self.create_training_summary()
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINING COMPLETE!")
        print("="*60)
        print(f"\nTrained {len(trained_models)} models successfully")
        print(f"Results saved to: {self.results_path}")
        
        return trained_models


def main():
    """Main execution function"""
    print("="*60)
    print("WASTE CLASSIFICATION - MODEL TRAINING")
    print("="*60)
    
    # Check if processed data exists
    results_path = Path('results')
    processed_data_path = results_path / 'processed_data'
    
    if not processed_data_path.exists():
        print("\n❌ Error: Processed data not found!")
        print("Please run data_preparation.py first to prepare the data.")
        return
    
    # Load processed data
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
    
    # Load data split info
    with open(results_path / 'metrics' / 'data_split_info.json', 'r') as f:
        data_info = json.load(f)
    
    class_names = data_info['class_names']
    num_classes = data_info['num_classes']
    img_size = tuple(data_info['image_size'])
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    # Initialize trainer
    trainer = WasteModelTrainer(
        results_path='results',
        img_shape=(img_size[0], img_size[1], 3),
        num_classes=num_classes,
        class_names=class_names
    )
    
    # Configure training parameters
    models_to_train = ['CustomCNN', 'DeepCNN', 'MobileNetV2', 'ResNet50', 'VGG16']
    epochs = 50
    batch_size = 32
    learning_rate = 0.001
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Models to train: {models_to_train}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Image size: {img_size}")
    
    # Ask for confirmation
    response = input("\nStart training? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Training cancelled.")
        return
    
    # Train all models
    trained_models = trainer.train_all_models(
        X_train, y_train,
        X_val, y_val,
        models_to_train=models_to_train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nSuccessfully trained {len(trained_models)} models:")
    for model_name in trained_models.keys():
        print(f"  ✓ {model_name}")
    
    print(f"\nAll results saved to: {results_path}")
    print("\nNext steps:")
    print("  1. Run evaluate.py to evaluate models on test data")
    print("  2. Run ensemble.py to create ensemble models")
    print("  3. Run predict.py to make predictions on new images")


if __name__ == "__main__":
    main()