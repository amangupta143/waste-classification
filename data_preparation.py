import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import json
from tqdm import tqdm
import cv2

class WasteDataPreparation:
    """
    Handles data loading, preprocessing, augmentation, and splitting
    for waste classification dataset
    """
    
    def __init__(self, dataset_path='dataset', results_path='results', 
                 img_size=(224, 224), test_size=0.2, val_size=0.1, random_state=42):
        """
        Initialize data preparation
        
        Args:
            dataset_path: Path to dataset folder containing category subfolders
            results_path: Path to store results
            img_size: Target image size (height, width)
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
        """
        self.dataset_path = Path(dataset_path)
        self.results_path = Path(results_path)
        self.img_size = img_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Create results directories
        self.create_directories()
        
        # Class names (sorted for consistency)
        self.class_names = sorted([d.name for d in self.dataset_path.iterdir() if d.is_dir()])
        self.num_classes = len(self.class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        print(f"Found {self.num_classes} classes: {self.class_names}")
        
    def create_directories(self):
        """Create necessary directories for storing results"""
        directories = [
            self.results_path,
            self.results_path / 'models',
            self.results_path / 'plots',
            self.results_path / 'metrics',
            self.results_path / 'reports'
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        print(f"Created results directories in: {self.results_path}")
    
    def load_data_info(self):
        """
        Scan dataset and collect information about images
        
        Returns:
            DataFrame with image paths, labels, and class indices
        """
        print("\nScanning dataset...")
        data = []
        
        for class_name in self.class_names:
            class_path = self.dataset_path / class_name
            image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + list(class_path.glob('*.png'))
            
            for img_path in image_files:
                data.append({
                    'image_path': str(img_path),
                    'class_name': class_name,
                    'class_idx': self.class_to_idx[class_name]
                })
        
        df = pd.DataFrame(data)
        print(f"Total images found: {len(df)}")
        
        # Display class distribution
        print("\nClass Distribution:")
        class_counts = df['class_name'].value_counts()
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images")
        
        # Save class distribution
        class_counts.to_csv(self.results_path / 'metrics' / 'class_distribution.csv')
        
        # Plot class distribution
        self.plot_class_distribution(class_counts)
        
        return df
    
    def plot_class_distribution(self, class_counts):
        """Plot and save class distribution"""
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(class_counts)), class_counts.values, color='skyblue', edgecolor='white', linewidth=1.5)
        plt.xlabel('Waste Category', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
        plt.xticks(range(len(class_counts)), class_counts.index, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_path / 'plots' / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Class distribution plot saved to: {self.results_path / 'plots' / 'class_distribution.png'}")
    
    def load_images(self, df):
        """
        Load and preprocess all images (WITHOUT normalization for generator compatibility)
        
        Args:
            df: DataFrame with image information
            
        Returns:
            X: Image array (num_samples, height, width, channels) - UNNORMALIZED [0-255]
            y: Label array (num_samples,)
        """
        print("\nLoading and preprocessing images...")
        X = []
        y = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
            try:
                # Load image
                img = load_img(row['image_path'], target_size=self.img_size)
                img_array = img_to_array(img)
                
                # DO NOT normalize here - let ImageDataGenerator handle it
                # This is crucial for proper augmentation
                
                X.append(img_array)
                y.append(row['class_idx'])
            except Exception as e:
                print(f"Error loading {row['image_path']}: {e}")
                continue
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        print(f"Loaded {len(X)} images with shape: {X.shape}")
        print(f"Pixel value range: [{X.min():.1f}, {X.max():.1f}]")
        
        return X, y
    
    def split_data(self, X, y):
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Image array
            y: Label array
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("\nSplitting data...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.val_size, random_state=self.random_state, stratify=y_temp
        )
        
        print(f"Training set: {len(X_train)} images ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Validation set: {len(X_val)} images ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Test set: {len(X_test)} images ({len(X_test)/len(X)*100:.1f}%)")
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
        
        # Save split information
        split_info = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'test_split_ratio': self.test_size,
            'val_split_ratio': self.val_size,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'image_size': self.img_size
        }
        
        with open(self.results_path / 'metrics' / 'data_split_info.json', 'w') as f:
            json.dump(split_info, f, indent=4)
        
        return X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, y_train, y_val, y_test
    
    def create_data_generators(self):
        """
        Create ImageDataGenerator for data augmentation and preprocessing
        
        Returns:
            train_generator, val_generator
        """
        # Training data generator with STRONG augmentation (inspired by best practices)
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # Normalize to [0,1]
            rotation_range=30,  # Increased rotation
            width_shift_range=0.25,  # Increased shift
            height_shift_range=0.25,
            shear_range=0.25,  # Increased shear
            zoom_range=0.25,  # Increased zoom
            horizontal_flip=True,
            vertical_flip=False,  # Not suitable for waste classification
            fill_mode='nearest',
            brightness_range=[0.7, 1.3]  # Wider brightness range
        )
        
        # Validation/Test data generator (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        print("\nData augmentation generators created")
        print("Training augmentation: rotation(30°), shift(0.25), shear(0.25), zoom(0.25), flip, brightness(0.7-1.3)")
        
        return train_datagen, val_datagen
    
    def visualize_samples(self, X, y, num_samples=16):
        """
        Visualize random samples from the dataset
        
        Args:
            X: Image array (UNNORMALIZED)
            y: Label array
            num_samples: Number of samples to visualize
        """
        print("\nCreating sample visualization...")
        
        # Select random samples
        indices = np.random.choice(len(X), size=min(num_samples, len(X)), replace=False)
        
        # Create grid
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx, ax in enumerate(axes):
            if idx < len(indices):
                img_idx = indices[idx]
                # Normalize for display
                img_display = X[img_idx] / 255.0
                ax.imshow(img_display)
                label_idx = np.argmax(y[img_idx]) if len(y[img_idx].shape) > 0 else y[img_idx]
                ax.set_title(self.class_names[label_idx], fontsize=10, fontweight='bold')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'plots' / 'sample_images.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sample images saved to: {self.results_path / 'plots' / 'sample_images.png'}")
    
    def visualize_augmentation(self, X, y, num_augmentations=5):
        """
        Visualize augmented versions of a sample image
        
        Args:
            X: Image array (UNNORMALIZED)
            y: Label array
            num_augmentations: Number of augmented versions to show
        """
        print("\nCreating augmentation visualization...")
        
        # Create augmentation generator with rescaling
        aug_datagen = ImageDataGenerator(
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
        
        # Select random image
        idx = np.random.randint(0, len(X))
        img = X[idx:idx+1]  # Keep batch dimension
        label_idx = np.argmax(y[idx]) if len(y[idx].shape) > 0 else y[idx]
        
        # Create augmented versions
        fig, axes = plt.subplots(1, num_augmentations + 1, figsize=(20, 4))
        
        # Original image (normalized for display)
        axes[0].imshow(X[idx] / 255.0)
        axes[0].set_title(f'Original\n{self.class_names[label_idx]}', fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        # Augmented images
        aug_iter = aug_datagen.flow(img, batch_size=1, seed=self.random_state)
        for i in range(num_augmentations):
            aug_img = next(aug_iter)[0]
            # Already normalized by generator
            aug_img = np.clip(aug_img, 0, 1)
            axes[i+1].imshow(aug_img)
            axes[i+1].set_title(f'Augmented {i+1}', fontsize=11, fontweight='bold')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'plots' / 'augmentation_examples.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Augmentation examples saved to: {self.results_path / 'plots' / 'augmentation_examples.png'}")
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Save preprocessed data to disk
        
        Args:
            X_train, X_val, X_test: Image arrays
            y_train, y_val, y_test: Label arrays
        """
        print("\nSaving preprocessed data...")
        
        data_path = self.results_path / 'processed_data'
        data_path.mkdir(exist_ok=True)
        
        np.save(data_path / 'X_train.npy', X_train)
        np.save(data_path / 'X_val.npy', X_val)
        np.save(data_path / 'X_test.npy', X_test)
        np.save(data_path / 'y_train.npy', y_train)
        np.save(data_path / 'y_val.npy', y_val)
        np.save(data_path / 'y_test.npy', y_test)
        
        print(f"Preprocessed data saved to: {data_path}")
    
    def prepare_data(self, save_processed=True):
        """
        Complete data preparation pipeline
        
        Args:
            save_processed: Whether to save preprocessed data
            
        Returns:
            Dictionary containing all prepared data
        """
        print("="*60)
        print("WASTE CLASSIFICATION - DATA PREPARATION")
        print("="*60)
        
        # Load data info
        df = self.load_data_info()
        
        # Load images
        X, y = self.load_images(df)
        
        # Split data
        X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, y_train, y_val, y_test = self.split_data(X, y)
        
        # Create data generators
        train_datagen, val_datagen = self.create_data_generators()
        
        # Visualizations
        self.visualize_samples(X_train, y_train_cat, num_samples=16)
        self.visualize_augmentation(X_train, y_train_cat, num_augmentations=5)
        
        # Save processed data
        if save_processed:
            self.save_processed_data(X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat)
        
        print("\n" + "="*60)
        print("DATA PREPARATION COMPLETE!")
        print("="*60)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train_cat,
            'y_val': y_val_cat,
            'y_test': y_test_cat,
            'y_train_labels': y_train,
            'y_val_labels': y_val,
            'y_test_labels': y_test,
            'train_datagen': train_datagen,
            'val_datagen': val_datagen,
            'class_names': self.class_names,
            'num_classes': self.num_classes
        }


def main():
    """Main execution function"""
    # Initialize data preparation
    data_prep = WasteDataPreparation(
        dataset_path='dataset',
        results_path='results',
        img_size=(224, 224),
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Prepare data
    data = data_prep.prepare_data(save_processed=True)
    
    print("\nData preparation summary:")
    print(f"  Training samples: {len(data['X_train'])}")
    print(f"  Validation samples: {len(data['X_val'])}")
    print(f"  Test samples: {len(data['X_test'])}")
    print(f"  Number of classes: {data['num_classes']}")
    print(f"  Image shape: {data['X_train'][0].shape}")
    print(f"\nAll results saved to: results/")
    print(f"\nNext step: Run train_models.py to train the models")


if __name__ == "__main__":
    main()