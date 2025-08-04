<<<<<<< HEAD
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter

class ImprovedASLModelTrainer:
    def __init__(self, data_dir="asl_data"):
        self.data_dir = data_dir
        self.gestures = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and preprocess data with improved error handling"""
        print("📥 Loading collected data...")
        
        X = []
        y = []
        total_samples = 0
        class_counts = {}
        
        for gesture in self.gestures:
            gesture_path = os.path.join(self.data_dir, gesture)
            
            if not os.path.exists(gesture_path):
                print(f"⚠️  No data found for gesture '{gesture}'")
                continue
            
            files = glob.glob(os.path.join(gesture_path, "*.npy"))
            
            if len(files) == 0:
                print(f"⚠️  No data files found for gesture '{gesture}'")
                continue
            
            gesture_samples = 0
            for file in files:
                try:
                    features = np.load(file)
                    
                    # Validate feature dimensions
                    if len(features) == 42:
                        # Data validation - check for NaN or infinite values
                        if not np.isnan(features).any() and not np.isinf(features).any():
                            X.append(features)
                            y.append(gesture)
                            gesture_samples += 1
                            total_samples += 1
                        else:
                            print(f"⚠️  Invalid data in {file} - contains NaN/Inf values")
                    else:
                        print(f"⚠️  Skipping {file} - wrong feature size: {len(features)}")
                        
                except Exception as e:
                    print(f"⚠️  Error loading {file}: {e}")
            
            class_counts[gesture] = gesture_samples
            print(f"Loaded {gesture_samples} samples for gesture '{gesture}'")
        
        if total_samples == 0:
            print("❌ No valid data found! Please collect data first.")
            return None, None, None
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Normalize features for better training
        X_normalized = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"✅ Loaded {total_samples} total samples")
        print(f"   Features shape: {X_normalized.shape}")
        print(f"   Labels shape: {y_encoded.shape}")
        
        # Check class balance
        self.check_class_balance(class_counts)
        
        return X_normalized, y_encoded, class_counts
    
    def check_class_balance(self, class_counts):
        """Check and report class imbalance"""
        counts = list(class_counts.values())
        min_count = min(counts)
        max_count = max(counts)
        
        print(f"\n📊 Class Balance Analysis:")
        print(f"   Min samples per class: {min_count}")
        print(f"   Max samples per class: {max_count}")
        print(f"   Imbalance ratio: {max_count/min_count if min_count > 0 else 'Infinite':.2f}")
        
        if max_count / min_count > 3:
            print("⚠️  Significant class imbalance detected! Consider collecting more data for underrepresented classes.")
    
    def augment_data(self, X, y):
        """Simple data augmentation with noise injection"""
        print("🔄 Applying data augmentation...")
        
        X_augmented = []
        y_augmented = []
        
        # Original data
        X_augmented.extend(X)
        y_augmented.extend(y)
        
        # Add slightly noisy versions
        for i in range(len(X)):
            # Small random noise
            noise = np.random.normal(0, 0.01, X[i].shape)
            X_noisy = X[i] + noise
            X_augmented.append(X_noisy)
            y_augmented.append(y[i])
            
            # Small scaling variations
            scale = np.random.uniform(0.95, 1.05)
            X_scaled = X[i] * scale
            X_augmented.append(X_scaled)
            y_augmented.append(y[i])
        
        print(f"✅ Data augmented: {len(X)} → {len(X_augmented)} samples")
        return np.array(X_augmented), np.array(y_augmented)
    
    def create_improved_model(self, input_shape, num_classes):
        """Create an improved neural network with better architecture"""
        print("🧠 Creating improved neural network model...")
        
        # Build model with batch normalization and better regularization
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(input_shape,)),
            
            # First hidden layer
            layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            # Second hidden layer
            layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            # Third hidden layer
            layers.Dense(64, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            
            # Fourth hidden layer
            layers.Dense(32),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.1),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Use a more sophisticated optimizer with learning rate scheduling
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Improved model created successfully!")
        print(f"   Input features: {input_shape}")
        print(f"   Output classes: {num_classes}")
        model.summary()
        
        return model
    
    def train_improved_model(self, X, y, test_size=0.2, epochs=100, batch_size=16, use_augmentation=True):
        """Train the improved model with better techniques"""
        print("🚀 Starting improved model training...")
        
        # Apply data augmentation if requested
        if use_augmentation:
            X, y = self.augment_data(X, y)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Create improved model
        self.model = self.create_improved_model(X.shape[1], len(self.gestures))
        
        # Enhanced callbacks
        callbacks = [
            # Early stopping with more patience
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpointing
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train with class weights to handle imbalance
        class_weights = self.calculate_class_weights(y_train)
        
        print("🏋️ Training in progress...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Load best model
        self.model = keras.models.load_model('best_model.h5')
        
        # Comprehensive evaluation
        self.comprehensive_evaluation(X_train, y_train, X_test, y_test)
        
        # Plot results
        self.plot_training_history(history)
        self.plot_confusion_matrix(y_test, self.model.predict(X_test))
        
        return history
    
    def calculate_class_weights(self, y_train):
        """Calculate class weights for imbalanced data"""
        class_counts = Counter(y_train)
        total_samples = len(y_train)
        num_classes = len(class_counts)
        
        class_weights = {}
        for class_id, count in class_counts.items():
            class_weights[class_id] = total_samples / (num_classes * count)
        
        print(f"📊 Using class weights to handle imbalance")
        return class_weights
    
    def comprehensive_evaluation(self, X_train, y_train, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n📊 Comprehensive Model Evaluation:")
        
        # Training accuracy
        train_predictions = self.model.predict(X_train, verbose=0)
        train_pred_classes = np.argmax(train_predictions, axis=1)
        train_accuracy = accuracy_score(y_train, train_pred_classes)
        
        # Testing accuracy
        test_predictions = self.model.predict(X_test, verbose=0)
        test_pred_classes = np.argmax(test_predictions, axis=1)
        test_accuracy = accuracy_score(y_test, test_pred_classes)
        
        print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Per-class accuracy
        print("\n📈 Per-Class Performance:")
        target_names = self.label_encoder.classes_
        report = classification_report(y_test, test_pred_classes, 
                                     target_names=target_names, 
                                     output_dict=True)
        
        for gesture in target_names:
            if gesture in report:
                precision = report[gesture]['precision']
                recall = report[gesture]['recall']
                f1 = report[gesture]['f1-score']
                print(f"   {gesture}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        # Confidence analysis
        self.analyze_prediction_confidence(test_predictions, y_test)
        
        return test_accuracy
    
    def analyze_prediction_confidence(self, predictions, y_true):
        """Analyze prediction confidence for faster recognition"""
        print("\n🎯 Prediction Confidence Analysis:")
        
        max_confidences = np.max(predictions, axis=1)
        correct_predictions = (np.argmax(predictions, axis=1) == y_true)
        
        # Confidence statistics for correct predictions
        correct_confidences = max_confidences[correct_predictions]
        incorrect_confidences = max_confidences[~correct_predictions]
        
        print(f"   Correct predictions - Mean confidence: {np.mean(correct_confidences):.3f}")
        print(f"   Incorrect predictions - Mean confidence: {np.mean(incorrect_confidences):.3f}")
        
        # Find optimal confidence threshold
        thresholds = np.arange(0.5, 1.0, 0.05)
        best_threshold = 0.8
        best_accuracy = 0
        
        for threshold in thresholds:
            high_conf_mask = max_confidences >= threshold
            if np.sum(high_conf_mask) > 0:
                high_conf_accuracy = np.mean(correct_predictions[high_conf_mask])
                coverage = np.mean(high_conf_mask)
                
                if high_conf_accuracy > best_accuracy and coverage > 0.7:
                    best_accuracy = high_conf_accuracy
                    best_threshold = threshold
        
        print(f"   Recommended confidence threshold: {best_threshold:.2f}")
        print(f"   Expected accuracy at this threshold: {best_accuracy:.3f}")
    
    def create_optimized_model_for_inference(self):
        """Create a quantized model for faster inference"""
        if self.model is None:
            print("❌ No trained model available for optimization!")
            return
        
        print("⚡ Creating optimized model for faster inference...")
        
        # Convert to TensorFlow Lite for faster inference
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Quantize the model
        tflite_model = converter.convert()
        
        # Save the optimized model
        with open('asl_model_optimized.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print("✅ Optimized TensorFlow Lite model saved as 'asl_model_optimized.tflite'")
        print("   This model will provide faster inference for real-time recognition!")
    
    def plot_training_history(self, history):
        """Enhanced training history visualization"""
        plt.figure(figsize=(15, 5))
        
        # Accuracy plot
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss plot
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        if 'lr' in history.history:
            plt.subplot(1, 3, 3)
            plt.plot(history.history['lr'], linewidth=2)
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Improved training history saved as 'improved_training_history.png'")
    
    def plot_confusion_matrix(self, y_true, predictions):
        """Enhanced confusion matrix with better visualization"""
        y_pred = np.argmax(predictions, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.label_encoder.classes_, 
                   yticklabels=self.label_encoder.classes_,
                   cbar_kws={'label': 'Normalized Frequency'})
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('improved_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Improved confusion matrix saved as 'improved_confusion_matrix.png'")
    
    def save_complete_model(self):
        """Save all model components"""
        if self.model is None:
            print("❌ No model to save! Train a model first.")
            return
        
        # Save Keras model
        self.model.save('improved_asl_model.h5')
        print("✅ Keras model saved as 'improved_asl_model.h5'")
        
        # Save label encoder
        with open('improved_label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print("✅ Label encoder saved as 'improved_label_encoder.pkl'")
        
        # Save scaler
        with open('feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("✅ Feature scaler saved as 'feature_scaler.pkl'")
        
        # Create optimized model for inference
        self.create_optimized_model_for_inference()
        
        print("\n🎯 Complete improved model training finished!")
        print("Files saved:")
        print("  - improved_asl_model.h5 (main model)")
        print("  - asl_model_optimized.tflite (fast inference)")
        print("  - improved_label_encoder.pkl (label mapping)")
        print("  - feature_scaler.pkl (data normalization)")

def main():
    """Main function with improved training pipeline"""
    print("🤖 Improved ASL Model Training System")
    print("=" * 50)
    
    trainer = ImprovedASLModelTrainer()
    
    # Load and preprocess data
    X, y, class_counts = trainer.load_data()
    
    if X is None:
        print("❌ Cannot proceed without data!")
        return
    
    # Training configuration
    print("\n⚙️  Enhanced Training Configuration:")
    epochs = int(input("Number of epochs (default 100): ") or 100)
    batch_size = int(input("Batch size (default 16): ") or 16)
    use_augmentation = input("Use data augmentation? (y/n, default y): ").lower() != 'n'
    
    # Train improved model
    print(f"\n🚀 Starting training with {epochs} epochs, batch size {batch_size}")
    history = trainer.train_improved_model(
        X, y, 
        epochs=epochs, 
        batch_size=batch_size,
        use_augmentation=use_augmentation
    )
    
    # Save everything
    trainer.save_complete_model()
    
    print("\n🎉 Training Complete!")
    print("Your improved ASL model should now have:")
    print("  - Higher accuracy (target: >95%)")
    print("  - Faster inference with TensorFlow Lite")
    print("  - Better confidence predictions")
    print("  - Reduced recognition time")

if __name__ == "__main__":
=======
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

class ASLModelTrainer:
    def __init__(self, data_dir="asl_data"):
        self.data_dir = data_dir
        self.gestures = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load all collected data"""
        print("📥 Loading collected data...")
        
        X = []  # Features (hand landmarks)
        y = []  # Labels (gesture letters)
        
        total_samples = 0
        
        for gesture in self.gestures:
            gesture_path = os.path.join(self.data_dir, gesture)
            
            if not os.path.exists(gesture_path):
                print(f"⚠️  No data found for gesture '{gesture}'")
                continue
            
            # Get all .npy files for this gesture
            files = glob.glob(os.path.join(gesture_path, "*.npy"))
            
            if len(files) == 0:
                print(f"⚠️  No data files found for gesture '{gesture}'")
                continue
            
            print(f"Loading {len(files)} samples for gesture '{gesture}'...")
            
            for file in files:
                try:
                    # Load the feature vector
                    features = np.load(file)
                    
                    # Ensure features are the right size (42 features)
                    if len(features) == 42:
                        X.append(features)
                        y.append(gesture)
                        total_samples += 1
                    else:
                        print(f"⚠️  Skipping {file} - wrong feature size: {len(features)}")
                        
                except Exception as e:
                    print(f"⚠️  Error loading {file}: {e}")
        
        if total_samples == 0:
            print("❌ No valid data found! Please collect data first.")
            return None, None
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Loaded {total_samples} total samples")
        print(f"   Features shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        
        # Encode labels to numbers
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X, y_encoded
    
    def create_model(self, input_shape, num_classes):
        """Create neural network model"""
        print("🧠 Creating neural network model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(input_shape,)),
            
            # Hidden layers with dropout for regularization
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model created successfully!")
        print(f"   Input features: {input_shape}")
        print(f"   Output classes: {num_classes}")
        
        return model
    
    def train_model(self, X, y, test_size=0.2, epochs=50, batch_size=32):
        """Train the model"""
        print("🚀 Starting model training...")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Create model
        self.model = self.create_model(X.shape[1], len(self.gestures))
        
        # Add early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        print("🏋️ Training in progress...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        print("\n📊 Evaluating model...")
        train_accuracy = self.model.evaluate(X_train, y_train, verbose=0)[1]
        test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)[1]
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")
        
        # Generate predictions for detailed evaluation
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Print classification report
        print("\n📈 Detailed Results:")
        target_names = self.label_encoder.classes_
        print(classification_report(y_test, y_pred_classes, target_names=target_names))
        
        # Plot training history
        self.plot_training_history(history)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred_classes, target_names)
        
        return history, test_accuracy
    
    def plot_training_history(self, history):
        """Plot training and validation curves"""
        plt.figure(figsize=(12, 4))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Training history saved as 'training_history.png'")
    
    def plot_confusion_matrix(self, y_true, y_pred, target_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Confusion matrix saved as 'confusion_matrix.png'")
    
    def save_model(self, model_filename="asl_model.h5", encoder_filename="label_encoder.pkl"):
        """Save trained model and label encoder"""
        if self.model is None:
            print("❌ No model to save! Train a model first.")
            return
        
        # Save model
        self.model.save(model_filename)
        print(f"✅ Model saved as '{model_filename}'")
        
        # Save label encoder
        with open(encoder_filename, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✅ Label encoder saved as '{encoder_filename}'")
        
        print("\n🎯 Model training complete!")
        print("Next step: Run the real-time prediction system!")

def main():
    """Main training function"""
    print("🤖 ASL Model Training System")
    print("=" * 40)
    
    trainer = ASLModelTrainer()
    
    # Load data
    X, y = trainer.load_data()
    
    if X is None:
        print("❌ Cannot proceed without data!")
        return
    
    # Show data summary
    unique_labels, counts = np.unique(y, return_counts=True)
    print("\n📊 Data Summary:")
    for label, count in zip(unique_labels, counts):
        gesture_name = trainer.label_encoder.classes_[label]
        print(f"   {gesture_name}: {count} samples")
    
    # Ask user for training parameters
    print("\n⚙️  Training Configuration:")
    epochs = int(input("Number of epochs (default 50): ") or 50)
    batch_size = int(input("Batch size (default 32): ") or 32)
    
    # Train model
    history, accuracy = trainer.train_model(X, y, epochs=epochs, batch_size=batch_size)
    
    # Save model
    trainer.save_model()
    
    print(f"\n🎉 Final Model Accuracy: {accuracy:.2%}")
    
    if accuracy > 0.85:
        print("🌟 Excellent accuracy! Your model is ready for real-time use!")
    elif accuracy > 0.70:
        print("👍 Good accuracy! Model should work well for most gestures.")
    else:
        print("⚠️  Lower accuracy. Consider collecting more data or adjusting parameters.")

if __name__ == "__main__":
>>>>>>> ef90c81ce9fa93d6e1cf089de58be96be22e2dfd
    main()