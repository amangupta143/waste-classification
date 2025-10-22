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
        Create a custom CNN architecture
        
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
        model.add(layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def create_transfer_learning_model(self, base_model_name='ResNet50', trainable_layers=0):
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
        
        # Freeze base model layers
        base_model.trainable = False
        if trainable_layers > 0:
            # Unfreeze the last N layers
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True
        
        # Create new model
        inputs = keras.Input(shape=self.img_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
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
    
    def get_callbacks(self, model_name, patience=10):
        """
        Create callbacks for training
        
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
        
        # Early stopping
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
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # CSV logger
        csv_path = self.metrics_path / f'{model_name}_training_log.csv'
        csv_logger = callbacks.CSVLogger(str(csv_path))
        
        return [checkpoint, early_stop, reduce_lr, csv_logger]
    
    def train_model(self, model, model_name, X_train, y_train, X_val, y_val, 
                   epochs=50, batch_size=32, train_datagen=None):
        """
        Train a single model
        
        Args:
            model: Keras model to train
            model_name: Name of the model
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            train_datagen: Data augmentation generator (optional)
            
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
        callback_list = self.get_callbacks(model_name, patience=10)
        
        # Record start time
        start_time = time.time()
        
        # Train with or without augmentation
        if train_datagen is not None:
            print(f"\nTraining with data augmentation...")
            history = model.fit(
                train_datagen.flow(X_train, y_train, batch_size=batch_size),
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callback_list,
                verbose=1
            )
        else:
            print(f"\nTraining without data augmentation...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
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
    
    def train_all_models(self, X_train, y_train, X_val, y_val, 
                        models_to_train=None, epochs=50, batch_size=32, 
                        train_datagen=None, learning_rate=0.001):
        """
        Train multiple models
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            models_to_train: List of model names to train
            epochs: Number of epochs
            batch_size: Batch size
            train_datagen: Data augmentation generator
            learning_rate: Learning rate
            
        Returns:
            Dictionary of trained models
        """
        if models_to_train is None:
            models_to_train = ['CustomCNN', 'ResNet50', 'VGG16', 'MobileNetV2', 'EfficientNetB0']
        
        print("\n" + "="*60)
        print("WASTE CLASSIFICATION - MODEL TRAINING")
        print("="*60)
        print(f"\nModels to train: {models_to_train}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        
        trained_models = {}
        
        for model_name in models_to_train:
            try:
                # Create model
                if model_name == 'CustomCNN':
                    model = self.create_custom_cnn(name=model_name)
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
                    X_train, y_train, X_val, y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    train_datagen=train_datagen
                )
                
                trained_models[model_name] = model
                
            except Exception as e:
                print(f"\nError training {model_name}: {str(e)}")
                continue
        
        # Create comparison summary
        self.create_training_summary()
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINING COMPLETE!")
        print("="*60)
        print(f"\nTrained {len(trained_models)} models successfully")
        print(f"Results saved to: {self.results_path}")
        
        return trained_models
    
    def create_training_summary(self):
        """Create a summary of all trained models"""
        if not self.training_histories:
            print("No training histories available")
            return
        
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
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('Best Val Accuracy', ascending=False)
        
        # Save to CSV
        summary_path = self.metrics_path / 'training_summary.csv'
        df_summary.to_csv(summary_path, index=False)
        
        print(f"\nTraining Summary:")
        print(df_summary.to_string(index=False))
        print(f"\nSummary saved to: {summary_path}")
        
        # Plot comparison
        self.plot_models_comparison(df_summary)
    
    def plot_models_comparison(self, df_summary):
        """Plot comparison of all models"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Best validation accuracy
        axes[0, 0].barh(df_summary['Model'], df_summary['Best Val Accuracy'], color='steelblue')
        axes[0, 0].set_xlabel('Accuracy')
        axes[0, 0].set_title('Best Validation Accuracy Comparison', fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Training time
        axes[0, 1].barh(df_summary['Model'], df_summary['Training Time (min)'], color='coral')
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_title('Training Time Comparison', fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # Accuracy difference (overfitting indicator)
        df_summary['Accuracy Gap'] = df_summary['Final Train Accuracy'] - df_summary['Final Val Accuracy']
        axes[1, 0].barh(df_summary['Model'], df_summary['Accuracy Gap'], color='lightgreen')
        axes[1, 0].set_xlabel('Accuracy Gap (Train - Val)')
        axes[1, 0].set_title('Overfitting Indicator (Lower is Better)', fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Epochs trained
        axes[1, 1].barh(df_summary['Model'], df_summary['Epochs Trained'], color='mediumpurple')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_title('Epochs Trained (Early Stopping)', fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'models_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Models comparison plot saved: {self.plots_path / 'models_comparison.png'}")


def main():
    """Main execution function"""
    # Load preprocessed data
    print("Loading preprocessed data...")
    data_path = Path('results/processed_data')
    
    X_train = np.load(data_path / 'X_train.npy')
    X_val = np.load(data_path / 'X_val.npy')
    y_train = np.load(data_path / 'y_train.npy')
    y_val = np.load(data_path / 'y_val.npy')
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Load data split info
    with open('results/metrics/data_split_info.json', 'r') as f:
        split_info = json.load(f)
    
    # Create data augmentation generator
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    
    # Initialize trainer
    trainer = WasteModelTrainer(
        results_path='results',
        img_shape=tuple(split_info['image_size']) + (3,),
        num_classes=split_info['num_classes'],
        class_names=split_info['class_names']
    )
    
    # Train all models
    models_to_train = ['CustomCNN', 'ResNet50', 'VGG16', 'MobileNetV2', 'EfficientNetB0']
    
    trained_models = trainer.train_all_models(
        X_train, y_train, X_val, y_val,
        models_to_train=models_to_train,
        epochs=50,
        batch_size=32,
        train_datagen=train_datagen,
        learning_rate=0.001
    )
    
    print(f"\nTraining complete! {len(trained_models)} models trained successfully.")


if __name__ == "__main__":
    main()