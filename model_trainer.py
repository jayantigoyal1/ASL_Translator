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
    main()