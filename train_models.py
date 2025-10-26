import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import shutil
from datetime import datetime
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import (
    ResNet50, VGG16, MobileNetV2, EfficientNetB0, InceptionV3
)
from tensorflow.keras.regularizers import l2

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Warning: ultralytics not installed. YOLOv8 models will be skipped.")


class WasteModelTrainer:
    """
    Enhanced trainer with optimized architectures and training strategies
    """
    
    def __init__(self, results_path='results', img_shape=(224, 224, 3), 
                 num_classes=9, class_names=None):
        """Initialize model trainer"""
        self.results_path = Path(results_path)
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        
        # Create directories
        self.models_path = self.results_path / 'models'
        self.plots_path = self.results_path / 'plots'
        self.metrics_path = self.results_path / 'metrics'
        self.yolo_data_path = self.results_path / 'yolo_split_dataset'
        
        for path in [self.models_path, self.plots_path, self.metrics_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        self.training_histories = {}
        
        print(f"Enhanced Model Trainer initialized")
        print(f"Image shape: {self.img_shape}")
        print(f"Number of classes: {self.num_classes}")
        print(f"YOLO available: {YOLO_AVAILABLE}")
    
    def create_custom_cnn_v1(self, name='CustomCNN_v1'):
        """Basic CNN - Simple and fast"""
        model = models.Sequential(name=name)
        
        # Block 1
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                input_shape=self.img_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Dropout(0.25))
        
        # Block 2
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Dropout(0.25))
        
        # Block 3
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Dropout(0.25))
        
        # Dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def create_deep_cnn_v2(self, name='DeepCNN_v2'):
        """Improved CNN with BatchNorm"""
        model = models.Sequential(name=name)
        
        # Block 1
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                input_shape=self.img_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Dropout(0.2))
        
        # Block 2
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Dropout(0.2))
        
        # Block 3
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Dropout(0.3))
        
        # Block 4
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Dropout(0.3))
        
        # Dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def create_residual_block(self, x, filters, kernel_size=3, stride=1, 
                             conv_shortcut=False, name=None):
        """Create a residual block with skip connections"""
        bn_axis = 3
        
        if conv_shortcut:
            shortcut = layers.Conv2D(filters, 1, strides=stride, 
                                    name=name + '_0_conv')(x)
            shortcut = layers.BatchNormalization(axis=bn_axis, 
                                                name=name + '_0_bn')(shortcut)
        else:
            shortcut = x
        
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                         name=name + '_1_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=name + '_1_bn')(x)
        x = layers.Activation('relu', name=name + '_1_relu')(x)
        
        x = layers.Conv2D(filters, kernel_size, padding='same',
                         name=name + '_2_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=name + '_2_bn')(x)
        
        x = layers.Add(name=name + '_add')([shortcut, x])
        x = layers.Activation('relu', name=name + '_out')(x)
        
        return x
    
    def create_deep_cnn_v3_fixed(self, name='DeepCNN_v3_Fixed'):
        """
        FIXED VERSION with Residual Connections
        Much more stable and better performance
        """
        inputs = keras.Input(shape=self.img_shape)
        
        # Initial conv
        x = layers.Conv2D(64, 7, strides=2, padding='same', name='conv1')(inputs)
        x = layers.BatchNormalization(name='bn_conv1')(x)
        x = layers.Activation('relu', name='relu_conv1')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)
        
        # Residual blocks - Stage 1 (64 filters)
        x = self.create_residual_block(x, 64, conv_shortcut=True, name='stage1_block1')
        x = self.create_residual_block(x, 64, name='stage1_block2')
        x = layers.Dropout(0.2)(x)
        
        # Residual blocks - Stage 2 (128 filters)
        x = self.create_residual_block(x, 128, stride=2, conv_shortcut=True, 
                                      name='stage2_block1')
        x = self.create_residual_block(x, 128, name='stage2_block2')
        x = layers.Dropout(0.3)(x)
        
        # Residual blocks - Stage 3 (256 filters)
        x = self.create_residual_block(x, 256, stride=2, conv_shortcut=True, 
                                      name='stage3_block1')
        x = self.create_residual_block(x, 256, name='stage3_block2')
        x = layers.Dropout(0.4)(x)
        
        # Global pooling and dense
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(512, activation='relu', name='fc1')(x)
        x = layers.BatchNormalization(name='bn_fc1')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu', name='fc2')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        model = keras.Model(inputs, outputs, name=name)
        
        return model
    
    def create_mobilenet_transfer(self, name='MobileNetV2_Transfer'):
        """MobileNetV2 with optimized transfer learning"""
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                 input_shape=self.img_shape)
        base_model.trainable = False  # Freeze initially
        
        # Improved head
        inputs = keras.Input(shape=self.img_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name=name)
        
        return model
    
    def create_transfer_learning_model(self, base_model_name='ResNet50', name=None):
        """Generic transfer learning with optimized architecture"""
        if name is None:
            name = f'{base_model_name}_Transfer'
        
        base_models = {
            'ResNet50': ResNet50,
            'VGG16': VGG16,
            'MobileNetV2': MobileNetV2,
            'EfficientNetB0': EfficientNetB0,
            'InceptionV3': InceptionV3
        }
        
        if base_model_name not in base_models:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        base_model = base_models[base_model_name](
            include_top=False,
            weights='imagenet',
            input_shape=self.img_shape
        )
        
        base_model.trainable = False
        
        # Optimized head
        inputs = keras.Input(shape=self.img_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs, name=name)
        
        return model
    
    def compile_model(self, model, learning_rate=0.001, use_adamw=False):
        """Compile model with optimizer, loss, and metrics"""
        if use_adamw:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=1e-4
            )
        else:
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        return model
    
    def get_callbacks(self, model_name, patience=10, use_cosine_annealing=False, epochs=100):
        """Create optimized callbacks for training"""
        checkpoint_path = self.models_path / f'{model_name}_best.h5'
        checkpoint = callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        callback_list = [checkpoint, early_stop]
        
        if use_cosine_annealing:
            # Cosine annealing schedule
            cosine_decay = callbacks.LearningRateScheduler(
                lambda epoch: 0.001 * (1 + np.cos(np.pi * epoch / epochs)) / 2,
                verbose=0
            )
            callback_list.append(cosine_decay)
        else:
            # Standard ReduceLROnPlateau
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
            callback_list.append(reduce_lr)
        
        csv_path = self.metrics_path / f'{model_name}_training_log.csv'
        csv_logger = callbacks.CSVLogger(str(csv_path))
        callback_list.append(csv_logger)
        
        return callback_list
    
    def progressive_unfreeze(self, model, base_model_name, unfreeze_layers=None):
        """
        Progressively unfreeze layers for fine-tuning
        
        Args:
            model: Keras model
            base_model_name: Name of base model
            unfreeze_layers: Number of layers to unfreeze from the end
        """
        # Find the base model layer
        base_model = None
        for layer in model.layers:
            if isinstance(layer, keras.Model):
                base_model = layer
                break
        
        if base_model is None:
            print("⚠️ Base model not found for unfreezing")
            return model
        
        # Default unfreezing strategy
        if unfreeze_layers is None:
            if base_model_name == 'MobileNetV2':
                unfreeze_layers = 30  # Unfreeze last 30 layers
            elif base_model_name in ['ResNet50', 'EfficientNetB0']:
                unfreeze_layers = 40
            elif base_model_name == 'VGG16':
                unfreeze_layers = 10
            else:
                unfreeze_layers = 20
        
        # Unfreeze
        base_model.trainable = True
        total_layers = len(base_model.layers)
        freeze_until = total_layers - unfreeze_layers
        
        for i, layer in enumerate(base_model.layers):
            if i < freeze_until:
                layer.trainable = False
            else:
                layer.trainable = True
        
        trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"\n🔓 Unfreezing: {trainable_count}/{total_layers} layers trainable")
        
        return model
    
    def train_keras_model(self, model, model_name, train_generator, val_data, 
                         epochs=100, steps_per_epoch=None, initial_lr=0.001,
                         use_adamw=False, use_cosine_annealing=False,
                         patience=10, fine_tune=False, base_model_name=None):
        """
        Enhanced training with optional fine-tuning phase
        """
        print("\n" + "="*60)
        print(f"Training {model_name}")
        print("="*60)
        
        # Phase 1: Initial training
        print(f"\n📚 PHASE 1: Initial Training")
        print(f"Learning rate: {initial_lr}")
        print(f"Patience: {patience}")
        
        callback_list = self.get_callbacks(model_name, patience=patience, 
                                          use_cosine_annealing=use_cosine_annealing, 
                                          epochs=epochs)
        
        start_time = time.time()
        
        history1 = model.fit(
            train_generator,
            validation_data=val_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callback_list,
            verbose=1
        )
        
        phase1_time = time.time() - start_time
        
        # Phase 2: Fine-tuning (for transfer learning models)
        if fine_tune and base_model_name:
            print(f"\n🎯 PHASE 2: Fine-Tuning with Progressive Unfreezing")
            
            # Load best model from phase 1
            best_model_path = self.models_path / f'{model_name}_best.h5'
            model = keras.models.load_model(str(best_model_path))
            
            # Unfreeze layers
            model = self.progressive_unfreeze(model, base_model_name)
            
            # Recompile with much lower learning rate
            fine_tune_lr = initial_lr / 10
            print(f"Fine-tuning learning rate: {fine_tune_lr}")
            
            model = self.compile_model(model, learning_rate=fine_tune_lr, 
                                      use_adamw=use_adamw)
            
            # Fine-tune callbacks
            ft_checkpoint = callbacks.ModelCheckpoint(
                str(self.models_path / f'{model_name}_finetuned_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            
            ft_early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            
            ft_reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-8,
                verbose=1
            )
            
            ft_csv_logger = callbacks.CSVLogger(
                str(self.metrics_path / f'{model_name}_finetuning_log.csv')
            )
            
            ft_callbacks = [ft_checkpoint, ft_early_stop, ft_reduce_lr, ft_csv_logger]
            
            # Fine-tune for fewer epochs
            ft_epochs = max(30, epochs // 2)
            
            start_time = time.time()
            
            history2 = model.fit(
                train_generator,
                validation_data=val_data,
                epochs=ft_epochs,
                steps_per_epoch=steps_per_epoch,
                callbacks=ft_callbacks,
                verbose=1
            )
            
            phase2_time = time.time() - start_time
            
            # Combine histories
            combined_history = {}
            for key in history1.history.keys():
                combined_history[key] = (history1.history[key] + 
                                        history2.history.get(key, []))
            
            total_time = phase1_time + phase2_time
            total_epochs = len(history1.history['loss']) + len(history2.history['loss'])
            
            print(f"\n✓ Total training time: {total_time/60:.2f} minutes")
            print(f"✓ Phase 1: {phase1_time/60:.2f} min ({len(history1.history['loss'])} epochs)")
            print(f"✓ Phase 2: {phase2_time/60:.2f} min ({len(history2.history['loss'])} epochs)")
            
            # Use fine-tuned model
            final_model_path = self.models_path / f'{model_name}_finetuned_best.h5'
            if final_model_path.exists():
                model.save(str(self.models_path / f'{model_name}_final.h5'))
        else:
            combined_history = history1.history
            total_time = phase1_time
            total_epochs = len(history1.history['loss'])
            
            # Save final model
            final_model_path = self.models_path / f'{model_name}_final.h5'
            model.save(str(final_model_path))
        
        # Store history
        self.training_histories[model_name] = {
            'history': combined_history,
            'training_time': total_time,
            'epochs_trained': total_epochs
        }
        
        # Save history to JSON
        history_data = {
            'model_name': model_name,
            'training_time_seconds': total_time,
            'epochs_trained': total_epochs,
            'history': {k: [float(v) for v in vals] 
                       for k, vals in combined_history.items()}
        }
        
        history_path = self.metrics_path / f'{model_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=4)
        
        print(f"\n{model_name} training completed!")
        print(f"Training time: {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {max(combined_history['val_accuracy']):.4f}")
        
        self.plot_training_history(model_name, combined_history)
        
        return model
    
    def prepare_yolo_dataset(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Prepare dataset in YOLO classification format"""
        print("\n" + "="*60)
        print("PREPARING YOLO DATASET FORMAT")
        print("="*60)
        
        if self.yolo_data_path.exists():
            shutil.rmtree(self.yolo_data_path)
        
        for split in ['train', 'val', 'test']:
            for class_name in self.class_names:
                split_path = self.yolo_data_path / split / class_name
                split_path.mkdir(parents=True, exist_ok=True)
        
        datasets = [
            ('train', X_train, y_train),
            ('val', X_val, y_val),
            ('test', X_test, y_test)
        ]
        
        for split_name, X, y in datasets:
            print(f"\nSaving {split_name} images...")
            for idx in tqdm(range(len(X)), desc=f"{split_name}"):
                class_idx = np.argmax(y[idx]) if len(y[idx].shape) > 0 else y[idx]
                class_name = self.class_names[class_idx]
                
                img = X[idx].astype(np.uint8)
                img_path = self.yolo_data_path / split_name / class_name / f'img_{idx}.jpg'
                
                from PIL import Image
                Image.fromarray(img).save(img_path)
        
        print(f"\n✓ YOLO dataset prepared at: {self.yolo_data_path}")
        
        return self.yolo_data_path
    
    def train_yolo_model(self, yolo_model_size='s', epochs=100, imgsz=640, batch=16):
        """Train YOLOv8 classification model"""
        if not YOLO_AVAILABLE:
            print("❌ YOLOv8 not available. Skipping YOLO training.")
            return None
        
        model_name = f'YOLOv8{yolo_model_size}'
        
        print("\n" + "="*60)
        print(f"Training {model_name} Classification")
        print("="*60)
        
        yolo_model_path = f'yolov8{yolo_model_size}-cls.pt'
        model = YOLO(yolo_model_path)
        
        start_time = time.time()
        
        print(f"\nTraining {model_name}...")
        results = model.train(
            data=str(self.yolo_data_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=15,  # Increased patience
            lr0=0.001,
            lrf=0.0001,  # Lower final LR
            weight_decay=0.0005,
            optimizer='AdamW',  # Use AdamW
            device=0 if tf.config.list_physical_devices('GPU') else 'cpu',
            augment=False,
            verbose=True,
            workers=2,
            cos_lr=True,  # Cosine annealing
            project=str(self.models_path / 'yolo_runs'),
            name=model_name,
            exist_ok=True
        )
        
        training_time = time.time() - start_time
        
        print(f"\nValidating {model_name} on test set...")
        metrics = model.val(data=str(self.yolo_data_path / 'test'), imgsz=imgsz)
        
        top1_acc = float(metrics.top1) if hasattr(metrics, 'top1') else 0.0
        top5_acc = float(metrics.top5) if hasattr(metrics, 'top5') else 0.0
        
        print(f"\n{model_name} Results:")
        print(f"  Top-1 Accuracy: {top1_acc:.4f}")
        print(f"  Top-5 Accuracy: {top5_acc:.4f}")
        print(f"  Training time: {training_time/60:.2f} minutes")
        
        # Save best model
        best_model_src = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
        best_model_dst = self.models_path / f'{model_name}_best.pt'
        if best_model_src.exists():
            shutil.copy(best_model_src, best_model_dst)
        
        # Plot results
        results_csv = Path(model.trainer.save_dir) / 'results.csv'
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            self.plot_yolo_history(model_name, df)
        
        self.training_histories[model_name] = {
            'history': {
                'accuracy': [],
                'val_accuracy': [top1_acc],
                'loss': [],
                'val_loss': []
            },
            'training_time': training_time,
            'epochs_trained': epochs,
            'top1_accuracy': top1_acc,
            'top5_accuracy': top5_acc
        }
        
        return results
    
    def plot_yolo_history(self, model_name, df):
        """Plot YOLO training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        if 'train/loss' in df.columns and 'val/loss' in df.columns:
            axes[0].plot(df['epoch'], df['train/loss'], label='Train Loss', linewidth=2)
            axes[0].plot(df['epoch'], df['val/loss'], label='Val Loss', linewidth=2)
            axes[0].set_title(f'{model_name} - Loss', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        if 'metrics/accuracy_top1' in df.columns:
            axes[1].plot(df['epoch'], df['metrics/accuracy_top1'], 
                        label='Top-1 Accuracy', linewidth=2)
            if 'metrics/accuracy_top5' in df.columns:
                axes[1].plot(df['epoch'], df['metrics/accuracy_top5'], 
                            label='Top-5 Accuracy', linewidth=2)
            axes[1].set_title(f'{model_name} - Accuracy', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / f'{model_name}_training_history.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(self, model_name, history):
        """Plot training and validation metrics"""
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
    
    def create_training_summary(self):
        """Create comprehensive summary of all trained models"""
        if not self.training_histories:
            print("No training histories available")
            return
        
        print("\n" + "="*60)
        print("CREATING TRAINING SUMMARY")
        print("="*60)
        
        summary_data = []
        for model_name, data in self.training_histories.items():
            history = data['history']
            
            if 'val_accuracy' in history and len(history['val_accuracy']) > 0:
                best_val_acc = max(history['val_accuracy'])
                final_val_acc = history['val_accuracy'][-1]
            elif 'top1_accuracy' in data:
                best_val_acc = data['top1_accuracy']
                final_val_acc = data['top1_accuracy']
            else:
                best_val_acc = 0.0
                final_val_acc = 0.0
            
            if 'accuracy' in history and len(history['accuracy']) > 0:
                final_train_acc = history['accuracy'][-1]
            else:
                final_train_acc = 0.0
            
            if 'loss' in history and len(history['loss']) > 0:
                final_train_loss = history['loss'][-1]
                final_val_loss = history['val_loss'][-1] if 'val_loss' in history else 0.0
                best_val_loss = min(history['val_loss']) if 'val_loss' in history else 0.0
            else:
                final_train_loss = 0.0
                final_val_loss = 0.0
                best_val_loss = 0.0
            
            summary_data.append({
                'Model': model_name,
                'Training Time (min)': data['training_time'] / 60,
                'Epochs Trained': data['epochs_trained'],
                'Final Train Accuracy': final_train_acc,
                'Final Val Accuracy': final_val_acc,
                'Best Val Accuracy': best_val_acc,
                'Final Train Loss': final_train_loss,
                'Final Val Loss': final_val_loss,
                'Best Val Loss': best_val_loss
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Best Val Accuracy', ascending=False)
        
        summary_df.to_csv(self.metrics_path / 'training_summary.csv', index=False)
        print(f"\n✓ Training summary saved")
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY - ALL MODELS")
        print("="*60)
        print(summary_df.to_string(index=False))
        
        self.plot_model_comparison(summary_df)
        
        return summary_df
    
    def plot_model_comparison(self, summary_df):
        """Create comparison plots for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models = summary_df['Model'].values
        
        axes[0, 0].barh(models, summary_df['Best Val Accuracy'], 
                       color='skyblue', edgecolor='navy')
        axes[0, 0].set_xlabel('Accuracy', fontsize=11)
        axes[0, 0].set_title('Best Validation Accuracy', fontsize=13, fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        axes[0, 1].barh(models, summary_df['Training Time (min)'], 
                       color='lightcoral', edgecolor='darkred')
        axes[0, 1].set_xlabel('Time (minutes)', fontsize=11)
        axes[0, 1].set_title('Training Time', fontsize=13, fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        axes[1, 0].barh(models, summary_df['Final Val Loss'], 
                       color='lightgreen', edgecolor='darkgreen')
        axes[1, 0].set_xlabel('Loss', fontsize=11)
        axes[1, 0].set_title('Final Validation Loss', fontsize=13, fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        axes[1, 1].barh(models, summary_df['Epochs Trained'], 
                       color='plum', edgecolor='purple')
        axes[1, 1].set_xlabel('Epochs', fontsize=11)
        axes[1, 1].set_title('Epochs Trained', fontsize=13, fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'models_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plot_accuracy_comparison()
    
    def plot_accuracy_comparison(self):
        """Plot validation accuracy comparison"""
        plt.figure(figsize=(14, 8))
        
        for model_name, data in self.training_histories.items():
            history = data['history']
            if 'val_accuracy' in history and len(history['val_accuracy']) > 0:
                epochs = range(1, len(history['val_accuracy']) + 1)
                plt.plot(epochs, history['val_accuracy'], 
                        label=model_name, linewidth=2, marker='o', markersize=4)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation Accuracy', fontsize=12)
        plt.title('Validation Accuracy Comparison - All Models', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_path / 'accuracy_comparison_all_models.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_all_models(self, X_train, y_train, X_val, y_val, X_test, y_test,
                        models_to_train=None, epochs=100, batch_size=32, 
                        train_datagen=None, val_datagen=None, yolo_config=None):
        """
        Train multiple models with optimized configurations
        """
        if models_to_train is None:
            models_to_train = [
                'CustomCNN_v1',
                'DeepCNN_v2', 
                'DeepCNN_v3_Fixed',  # Fixed version
                'MobileNetV2_Transfer',
                'ResNet50',
                'VGG16',
                'EfficientNetB0'
            ]
            if YOLO_AVAILABLE:
                models_to_train.extend(['YOLOv8n', 'YOLOv8s'])
        
        print("\n" + "="*60)
        print("OPTIMIZED WASTE CLASSIFICATION - MODEL TRAINING")
        print("="*60)
        print(f"\nModels to train: {models_to_train}")
        print(f"Epochs (Keras): {epochs}")
        print(f"Batch size: {batch_size}")
        
        # Create generators if needed
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
        
        # Prepare YOLO data
        yolo_models = [m for m in models_to_train if m.startswith('YOLO')]
        if yolo_models and YOLO_AVAILABLE:
            self.prepare_yolo_dataset(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Prepare Keras data
        train_generator = train_datagen.flow(X_train, y_train, 
                                            batch_size=batch_size, shuffle=True)
        
        X_val_normalized = val_datagen.flow(X_val, y_val, 
                                           batch_size=len(X_val), 
                                           shuffle=False).next()
        
        steps_per_epoch = len(X_train) // batch_size
        
        trained_models = {}
        
        # Model-specific configurations
        model_configs = {
            'CustomCNN_v1': {'lr': 0.001, 'patience': 10, 'fine_tune': False, 
                           'use_adamw': False, 'cosine': False},
            'DeepCNN_v2': {'lr': 0.001, 'patience': 10, 'fine_tune': False, 
                          'use_adamw': True, 'cosine': False},
            'DeepCNN_v3_Fixed': {'lr': 0.0001, 'patience': 12, 'fine_tune': False, 
                                'use_adamw': True, 'cosine': True},
            'MobileNetV2_Transfer': {'lr': 0.001, 'patience': 10, 'fine_tune': True, 
                                    'use_adamw': True, 'cosine': True},
            'ResNet50': {'lr': 0.0001, 'patience': 12, 'fine_tune': True, 
                        'use_adamw': True, 'cosine': True},
            'VGG16': {'lr': 0.0001, 'patience': 10, 'fine_tune': True, 
                     'use_adamw': True, 'cosine': False},
            'EfficientNetB0': {'lr': 0.0001, 'patience': 12, 'fine_tune': True, 
                              'use_adamw': True, 'cosine': True}
        }
        
        # Train each model
        for model_name in models_to_train:
            try:
                if model_name.startswith('YOLO') and YOLO_AVAILABLE:
                    yolo_size = model_name.replace('YOLOv8', '').lower()
                    yolo_cfg = yolo_config or {}
                    self.train_yolo_model(
                        yolo_model_size=yolo_size,
                        epochs=yolo_cfg.get('epochs', 100),
                        imgsz=yolo_cfg.get('imgsz', 640),
                        batch=yolo_cfg.get('batch', 16)
                    )
                    trained_models[model_name] = 'YOLO'
                
                elif model_name == 'CustomCNN_v1':
                    config = model_configs[model_name]
                    model = self.create_custom_cnn_v1(name=model_name)
                    model = self.compile_model(model, config['lr'], config['use_adamw'])
                    self.train_keras_model(
                        model, model_name, train_generator, X_val_normalized,
                        epochs, steps_per_epoch, config['lr'], config['use_adamw'],
                        config['cosine'], config['patience']
                    )
                    trained_models[model_name] = model
                
                elif model_name == 'DeepCNN_v2':
                    config = model_configs[model_name]
                    model = self.create_deep_cnn_v2(name=model_name)
                    model = self.compile_model(model, config['lr'], config['use_adamw'])
                    self.train_keras_model(
                        model, model_name, train_generator, X_val_normalized,
                        epochs, steps_per_epoch, config['lr'], config['use_adamw'],
                        config['cosine'], config['patience']
                    )
                    trained_models[model_name] = model
                
                elif model_name == 'DeepCNN_v3_Fixed':
                    config = model_configs[model_name]
                    model = self.create_deep_cnn_v3_fixed(name=model_name)
                    model = self.compile_model(model, config['lr'], config['use_adamw'])
                    self.train_keras_model(
                        model, model_name, train_generator, X_val_normalized,
                        epochs, steps_per_epoch, config['lr'], config['use_adamw'],
                        config['cosine'], config['patience']
                    )
                    trained_models[model_name] = model
                
                elif model_name == 'MobileNetV2_Transfer':
                    config = model_configs[model_name]
                    model = self.create_mobilenet_transfer(name=model_name)
                    model = self.compile_model(model, config['lr'], config['use_adamw'])
                    self.train_keras_model(
                        model, model_name, train_generator, X_val_normalized,
                        epochs, steps_per_epoch, config['lr'], config['use_adamw'],
                        config['cosine'], config['patience'], config['fine_tune'],
                        'MobileNetV2'
                    )
                    trained_models[model_name] = model
                
                elif model_name in ['ResNet50', 'VGG16', 'EfficientNetB0']:
                    config = model_configs[model_name]
                    model = self.create_transfer_learning_model(
                        base_model_name=model_name,
                        name=f'{model_name}_Transfer'
                    )
                    model = self.compile_model(model, config['lr'], config['use_adamw'])
                    self.train_keras_model(
                        model, f'{model_name}_Transfer', train_generator, 
                        X_val_normalized, epochs, steps_per_epoch, config['lr'],
                        config['use_adamw'], config['cosine'], config['patience'],
                        config['fine_tune'], model_name
                    )
                    trained_models[model_name] = model
                
                else:
                    print(f"\n⚠️  Unknown model: {model_name}, skipping...")
                    continue
                
            except Exception as e:
                print(f"\n❌ Error training {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        self.create_training_summary()
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINING COMPLETE!")
        print("="*60)
        
        return trained_models


def main():
    """Main execution function"""
    print("="*60)
    print("OPTIMIZED WASTE CLASSIFICATION - MODEL TRAINING")
    print("="*60)
    
    results_path = Path('results')
    processed_data_path = results_path / 'processed_data'
    
    if not processed_data_path.exists():
        print("\n❌ Error: Processed data not found!")
        print("Please run data_preparation.py first.")
        return
    
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
    
    # Configure training
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    
    models_to_train = [
        'CustomCNN_v1',
        'DeepCNN_v2',
        'DeepCNN_v3_Fixed',
        'MobileNetV2_Transfer',
        'ResNet50',
        'VGG16',
        'EfficientNetB0'
    ]
    
    if YOLO_AVAILABLE:
        models_to_train.extend(['YOLOv8n', 'YOLOv8s'])
    
    epochs = 100
    batch_size = 32
    
    yolo_config = {
        'epochs': 100,
        'imgsz': 640,
        'batch': 16
    }
    
    print(f"Models to train: {models_to_train}")
    print(f"Epochs (Keras): {epochs}")
    print(f"Epochs (YOLO): {yolo_config['epochs']}")
    print(f"Batch size (Keras): {batch_size}")
    print(f"Batch size (YOLO): {yolo_config['batch']}")
    print(f"Image size: {img_size}")
    
    print("\n" + "="*60)
    print("KEY IMPROVEMENTS IN THIS VERSION:")
    print("="*60)
    print("✓ Fixed DeepCNN_v3 with residual connections")
    print("✓ Lower learning rates for transfer learning (0.0001)")
    print("✓ Progressive unfreezing for fine-tuning")
    print("✓ AdamW optimizer with weight decay")
    print("✓ Cosine annealing LR schedules")
    print("✓ Increased patience (10-12 epochs)")
    print("✓ More epochs (100) for better convergence")
    print("✓ Two-phase training for transfer models")
    
    # Ask for confirmation
    response = input("\nStart training? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Training cancelled.")
        return
    
    # Train all models
    print("\n" + "="*60)
    print("STARTING TRAINING PIPELINE")
    print("="*60)
    
    trained_models = trainer.train_all_models(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        models_to_train=models_to_train,
        epochs=epochs,
        batch_size=batch_size,
        yolo_config=yolo_config
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\n✓ Successfully trained {len(trained_models)} models:")
    for model_name in trained_models.keys():
        print(f"  ✓ {model_name}")
    
    print(f"\n✓ All results saved to: {results_path}")
    
    # Display best model
    summary_csv = results_path / 'metrics' / 'training_summary.csv'
    if summary_csv.exists():
        summary_df = pd.read_csv(summary_csv)
        print("\n" + "="*60)
        print("TOP 5 MODELS BY VALIDATION ACCURACY")
        print("="*60)
        top5 = summary_df.head(5)
        for idx, row in top5.iterrows():
            print(f"\n{idx+1}. {row['Model']}")
            print(f"   Val Accuracy: {row['Best Val Accuracy']:.4f}")
            print(f"   Training Time: {row['Training Time (min)']:.2f} min")
            print(f"   Epochs: {int(row['Epochs Trained'])}")
        
        best_model = summary_df.iloc[0]
        print("\n" + "="*60)
        print(f"🏆 BEST MODEL: {best_model['Model']}")
        print("="*60)
        print(f"   Validation Accuracy: {best_model['Best Val Accuracy']:.4f}")
        print(f"   Training Time: {best_model['Training Time (min)']:.2f} minutes")
        print(f"   Final Val Loss: {best_model['Final Val Loss']:.4f}")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("  1. Run evaluate.py to test models on test set")
    print("  2. Run ensemble.py to create ensemble models")
    print("  3. Run predict.py to make predictions on new images")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()