import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image
import argparse

import tensorflow as tf
from tensorflow import keras

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Warning: ultralytics not installed. YOLO models will be skipped.")


class WasteClassificationPredictor:
    """
    Make predictions on new waste images using trained models
    """
    
    def __init__(self, results_path='results', model_name=None, model_type='keras'):
        """
        Initialize predictor
        
        Args:
            results_path: Path to results directory
            model_name: Name of the model to use (None = use best)
            model_type: 'keras', 'yolo', or 'ensemble'
        """
        self.results_path = Path(results_path)
        self.models_path = self.results_path / 'models'
        self.model_name = model_name
        self.model_type = model_type
        
        # Load class names
        data_info_path = self.results_path / 'metrics' / 'data_split_info.json'
        if data_info_path.exists():
            with open(data_info_path, 'r') as f:
                data_info = json.load(f)
            self.class_names = data_info['class_names']
            self.num_classes = data_info['num_classes']
            self.img_size = tuple(data_info['image_size'])
        else:
            raise FileNotFoundError("Data split info not found. Please run data_preparation.py first.")
        
        # Load model
        self.model = None
        self.load_model()
        
        print("="*60)
        print("WASTE CLASSIFICATION PREDICTOR")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Type: {self.model_type}")
        print(f"Classes: {self.num_classes}")
        print(f"Image size: {self.img_size}")
    
    def find_best_model(self):
        """Find the best performing model from training summary"""
        summary_path = self.results_path / 'metrics' / 'training_summary.csv'
        
        if summary_path.exists():
            import pandas as pd
            summary_df = pd.read_csv(summary_path)
            best_model = summary_df.iloc[0]['Model']
            
            # Determine model type
            if best_model.startswith('YOLO'):
                return best_model, 'yolo'
            else:
                return best_model, 'keras'
        
        # Fallback: find any available model
        keras_models = list(self.models_path.glob('*_best.h5'))
        if keras_models:
            model_name = keras_models[0].stem.replace('_best', '')
            return model_name, 'keras'
        
        yolo_models = list(self.models_path.glob('YOLO*_best.pt'))
        if yolo_models and YOLO_AVAILABLE:
            model_name = yolo_models[0].stem.replace('_best', '')
            return model_name, 'yolo'
        
        raise FileNotFoundError("No trained models found!")
    
    def load_model(self):
        """Load the specified model"""
        if self.model_name is None:
            self.model_name, self.model_type = self.find_best_model()
            print(f"Using best model: {self.model_name}")
        
        model_path = self.models_path / f'{self.model_name}_best.{"h5" if self.model_type == "keras" else "pt"}'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        if self.model_type == 'keras':
            self.model = keras.models.load_model(str(model_path))
        elif self.model_type == 'yolo':
            if not YOLO_AVAILABLE:
                raise ImportError("YOLO not available. Install ultralytics package.")
            self.model = YOLO(str(model_path))
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print("✓ Model loaded successfully")
    
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array and original image
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        original_img = np.array(img)
        
        # Resize
        img_resized = img.resize(self.img_size)
        img_array = np.array(img_resized)
        
        return img_array, original_img
    
    def predict_single(self, image_path, show_plot=True):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            show_plot: Whether to display the result
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess
        img_array, original_img = self.load_and_preprocess_image(image_path)
        
        # Predict
        if self.model_type == 'keras':
            # Normalize for Keras
            img_normalized = img_array / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            predictions = self.model.predict(img_batch, verbose=0)[0]
        
        elif self.model_type == 'yolo':
            # YOLO expects unnormalized image
            results = self.model(img_array, verbose=False)
            predictions = results[0].probs.data.cpu().numpy()
        
        # Get prediction results
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[predicted_class_idx]
        
        # Get top 5 predictions
        top5_indices = np.argsort(predictions)[-5:][::-1]
        top5_predictions = [
            {
                'class': self.class_names[idx],
                'confidence': float(predictions[idx]),
                'percentage': float(predictions[idx] * 100)
            }
            for idx in top5_indices
        ]
        
        result = {
            'image_path': str(image_path),
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'percentage': float(confidence * 100),
            'top5_predictions': top5_predictions,
            'all_probabilities': predictions.tolist()
        }
        
        # Display result
        if show_plot:
            self.plot_prediction(original_img, result)
        
        return result
    
    def predict_batch(self, image_paths, show_plot=False):
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image paths
            show_plot: Whether to display results
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"\nPredicting {len(image_paths)} images...")
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path, show_plot=False)
                results.append(result)
                print(f"✓ {Path(image_path).name}: {result['predicted_class']} ({result['percentage']:.1f}%)")
            except Exception as e:
                print(f"❌ Error processing {image_path}: {str(e)}")
                continue
        
        if show_plot and results:
            self.plot_batch_predictions(results)
        
        return results
    
    def predict_folder(self, folder_path, show_plot=False):
        """
        Predict classes for all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            show_plot: Whether to display results
            
        Returns:
            List of prediction results
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_paths = [
            f for f in folder.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            print(f"No images found in {folder_path}")
            return []
        
        print(f"\nFound {len(image_paths)} images in {folder_path}")
        
        return self.predict_batch(image_paths, show_plot=show_plot)
    
    def plot_prediction(self, image, result):
        """
        Plot single image prediction
        
        Args:
            image: Image array
            result: Prediction result dictionary
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Display image
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title(f'Predicted: {result["predicted_class"]}\n'
                     f'Confidence: {result["percentage"]:.1f}%',
                     fontsize=12, fontweight='bold')
        
        # Display top 5 predictions
        top5 = result['top5_predictions']
        classes = [p['class'] for p in top5]
        confidences = [p['confidence'] for p in top5]
        
        colors = ['green' if i == 0 else 'skyblue' for i in range(len(classes))]
        bars = ax2.barh(range(len(classes)), confidences, color=colors, edgecolor='navy')
        
        # Add percentage labels
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            ax2.text(conf + 0.01, i, f'{conf*100:.1f}%',
                    va='center', fontsize=10, fontweight='bold')
        
        ax2.set_yticks(range(len(classes)))
        ax2.set_yticklabels(classes)
        ax2.set_xlabel('Confidence', fontsize=11, fontweight='bold')
        ax2.set_title('Top 5 Predictions', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 1.1)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_batch_predictions(self, results):
        """
        Plot predictions for multiple images
        
        Args:
            results: List of prediction results
        """
        n_images = len(results)
        n_cols = min(4, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, result in enumerate(results):
            if idx >= len(axes):
                break
            
            # Load image
            img = Image.open(result['image_path'])
            
            # Display
            axes[idx].imshow(img)
            axes[idx].axis('off')
            
            title = f"{result['predicted_class']}\n{result['percentage']:.1f}%"
            axes[idx].set_title(title, fontsize=10, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_predictions(self, results, output_path='predictions.json'):
        """
        Save prediction results to JSON file
        
        Args:
            results: List of prediction results
            output_path: Path to save JSON file
        """
        output_file = self.results_path / output_path
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n✓ Predictions saved to: {output_file}")
    
    def generate_prediction_report(self, results):
        """
        Generate a comprehensive prediction report
        
        Args:
            results: List of prediction results
        """
        report_path = self.results_path / 'reports' / 'prediction_report.txt'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("WASTE CLASSIFICATION - PREDICTION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Model Used: {self.model_name}\n")
            f.write(f"Model Type: {self.model_type}\n")
            f.write(f"Total Images: {len(results)}\n\n")
            
            # Class distribution
            from collections import Counter
            predicted_classes = [r['predicted_class'] for r in results]
            class_counts = Counter(predicted_classes)
            
            f.write("PREDICTED CLASS DISTRIBUTION\n")
            f.write("-"*80 + "\n")
            for class_name, count in class_counts.most_common():
                percentage = count / len(results) * 100
                f.write(f"{class_name}: {count} ({percentage:.1f}%)\n")
            
            # Confidence statistics
            confidences = [r['confidence'] for r in results]
            f.write("\n" + "="*80 + "\n")
            f.write("CONFIDENCE STATISTICS\n")
            f.write("="*80 + "\n")
            f.write(f"Mean Confidence: {np.mean(confidences):.4f}\n")
            f.write(f"Median Confidence: {np.median(confidences):.4f}\n")
            f.write(f"Min Confidence: {np.min(confidences):.4f}\n")
            f.write(f"Max Confidence: {np.max(confidences):.4f}\n")
            f.write(f"Std Dev: {np.std(confidences):.4f}\n")
            
            # Low confidence predictions
            low_conf_threshold = 0.5
            low_conf_results = [r for r in results if r['confidence'] < low_conf_threshold]
            
            if low_conf_results:
                f.write("\n" + "="*80 + "\n")
                f.write(f"LOW CONFIDENCE PREDICTIONS (< {low_conf_threshold})\n")
                f.write("="*80 + "\n")
                for result in low_conf_results:
                    f.write(f"\n{Path(result['image_path']).name}:\n")
                    f.write(f"  Predicted: {result['predicted_class']}\n")
                    f.write(f"  Confidence: {result['confidence']:.4f}\n")
                    f.write(f"  Top 3 alternatives:\n")
                    for pred in result['top5_predictions'][1:4]:
                        f.write(f"    - {pred['class']}: {pred['percentage']:.1f}%\n")
            
            # Detailed predictions
            f.write("\n" + "="*80 + "\n")
            f.write("DETAILED PREDICTIONS\n")
            f.write("="*80 + "\n")
            
            for idx, result in enumerate(results, 1):
                f.write(f"\n{idx}. {Path(result['image_path']).name}\n")
                f.write(f"   Predicted Class: {result['predicted_class']}\n")
                f.write(f"   Confidence: {result['percentage']:.1f}%\n")
                f.write(f"   Top 5 Predictions:\n")
                for pred in result['top5_predictions']:
                    f.write(f"     {pred['class']}: {pred['percentage']:.1f}%\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"✓ Prediction report saved to: {report_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Predict waste categories for images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single image (using best model)
  python predict.py --image path/to/image.jpg
  
  # Predict multiple images
  python predict.py --images image1.jpg image2.jpg image3.jpg
  
  # Predict all images in a folder
  python predict.py --folder path/to/images/
  
  # Use specific model
  python predict.py --image test.jpg --model MobileNetV2_Transfer --type keras
  
  # Save predictions to JSON
  python predict.py --folder images/ --save predictions.json
        """
    )
    
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--images', nargs='+', help='Paths to multiple images')
    parser.add_argument('--folder', type=str, help='Path to folder containing images')
    parser.add_argument('--model', type=str, default=None, help='Model name (default: best model)')
    parser.add_argument('--type', type=str, default='keras', choices=['keras', 'yolo'],
                       help='Model type (default: keras)')
    parser.add_argument('--save', type=str, default=None, help='Save predictions to JSON file')
    parser.add_argument('--report', action='store_true', help='Generate prediction report')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--results-path', type=str, default='results', 
                       help='Path to results directory (default: results)')
    
    args = parser.parse_args()
    
    # Check if at least one input is provided
    if not any([args.image, args.images, args.folder]):
        print("="*60)
        print("WASTE CLASSIFICATION - PREDICTION")
        print("="*60)
        print("\n❌ Error: No input provided!")
        print("\nPlease specify one of:")
        print("  --image IMAGE_PATH       : Predict single image")
        print("  --images IMG1 IMG2 ...   : Predict multiple images")
        print("  --folder FOLDER_PATH     : Predict all images in folder")
        print("\nFor help: python predict.py --help")
        return
    
    # Initialize predictor
    try:
        predictor = WasteClassificationPredictor(
            results_path=args.results_path,
            model_name=args.model,
            model_type=args.type
        )
    except Exception as e:
        print(f"\n❌ Error initializing predictor: {str(e)}")
        return
    
    # Make predictions
    results = []
    show_plot = not args.no_plot
    
    try:
        if args.image:
            # Single image
            print("\n" + "="*60)
            print("PREDICTING SINGLE IMAGE")
            print("="*60)
            result = predictor.predict_single(args.image, show_plot=show_plot)
            results = [result]
            
            print(f"\n✓ Prediction: {result['predicted_class']}")
            print(f"  Confidence: {result['percentage']:.1f}%")
            print("\nTop 5 Predictions:")
            for i, pred in enumerate(result['top5_predictions'], 1):
                print(f"  {i}. {pred['class']}: {pred['percentage']:.1f}%")
        
        elif args.images:
            # Multiple images
            print("\n" + "="*60)
            print("PREDICTING MULTIPLE IMAGES")
            print("="*60)
            results = predictor.predict_batch(args.images, show_plot=show_plot)
        
        elif args.folder:
            # Folder
            print("\n" + "="*60)
            print("PREDICTING FOLDER")
            print("="*60)
            results = predictor.predict_folder(args.folder, show_plot=show_plot)
        
        # Summary
        if results:
            print("\n" + "="*60)
            print("PREDICTION SUMMARY")
            print("="*60)
            print(f"\nTotal images processed: {len(results)}")
            
            # Class distribution
            from collections import Counter
            predicted_classes = [r['predicted_class'] for r in results]
            class_counts = Counter(predicted_classes)
            
            print("\nPredicted classes:")
            for class_name, count in class_counts.most_common():
                percentage = count / len(results) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
            
            # Confidence statistics
            confidences = [r['confidence'] for r in results]
            print(f"\nConfidence statistics:")
            print(f"  Mean: {np.mean(confidences):.3f}")
            print(f"  Median: {np.median(confidences):.3f}")
            print(f"  Min: {np.min(confidences):.3f}")
            print(f"  Max: {np.max(confidences):.3f}")
            
            # Low confidence warnings
            low_conf = [r for r in results if r['confidence'] < 0.5]
            if low_conf:
                print(f"\n⚠️  {len(low_conf)} prediction(s) with low confidence (< 50%)")
                print("   Review these predictions manually:")
                for r in low_conf:
                    print(f"   • {Path(r['image_path']).name}: "
                          f"{r['predicted_class']} ({r['percentage']:.1f}%)")
        
        # Save predictions
        if args.save and results:
            predictor.save_predictions(results, args.save)
        
        # Generate report
        if args.report and results:
            print("\nGenerating prediction report...")
            predictor.generate_prediction_report(results)
        
        print("\n" + "="*60)
        print("PREDICTION COMPLETE!")
        print("="*60)
    
    except Exception as e:
        print(f"\n❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # If no arguments provided, run interactive mode
    import sys
    
    if len(sys.argv) == 1:
        print("="*60)
        print("WASTE CLASSIFICATION - PREDICTION (Interactive Mode)")
        print("="*60)
        
        try:
            # Initialize predictor
            predictor = WasteClassificationPredictor()
            
            print("\n" + "="*60)
            print("PREDICTION OPTIONS")
            print("="*60)
            print("\n1. Predict single image")
            print("2. Predict multiple images")
            print("3. Predict folder")
            print("4. Exit")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "1":
                image_path = input("Enter image path: ").strip()
                if Path(image_path).exists():
                    result = predictor.predict_single(image_path, show_plot=True)
                    print(f"\n✓ Prediction: {result['predicted_class']}")
                    print(f"  Confidence: {result['percentage']:.1f}%")
                else:
                    print(f"❌ Image not found: {image_path}")
            
            elif choice == "2":
                images_input = input("Enter image paths (comma-separated): ").strip()
                image_paths = [p.strip() for p in images_input.split(',')]
                results = predictor.predict_batch(image_paths, show_plot=True)
                
                if results:
                    save = input("\nSave predictions? (yes/no): ").strip().lower()
                    if save in ['yes', 'y']:
                        predictor.save_predictions(results, 'predictions.json')
            
            elif choice == "3":
                folder_path = input("Enter folder path: ").strip()
                if Path(folder_path).exists():
                    results = predictor.predict_folder(folder_path, show_plot=False)
                    
                    if results:
                        print(f"\n✓ Processed {len(results)} images")
                        
                        save = input("\nSave predictions? (yes/no): ").strip().lower()
                        if save in ['yes', 'y']:
                            predictor.save_predictions(results, 'predictions.json')
                        
                        report = input("Generate report? (yes/no): ").strip().lower()
                        if report in ['yes', 'y']:
                            predictor.generate_prediction_report(results)
                else:
                    print(f"❌ Folder not found: {folder_path}")
            
            elif choice == "4":
                print("Exiting...")
            
            else:
                print("Invalid option")
        
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    else:
        # Command-line mode
        main()