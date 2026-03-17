import tensorflow as tf
from data_loader import get_data
from models import build_unmasking_model, build_emotion_model
import os

# Create models directory
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# 1. Load Data
print("Loading data...")
(X_train_masked, X_train_orig, y_train), (X_test_masked, X_test_orig, y_test) = get_data(num_samples=2000)

# 2. Train Unmasking Autoencoder
print("\n--- Training Unmasking Model ---")
autoencoder = build_unmasking_model()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train on MASKED input -> ORIGINAL target
autoencoder.fit(X_train_masked, X_train_orig,
                epochs=10,
                batch_size=32,
                validation_data=(X_test_masked, X_test_orig))

autoencoder.save('saved_models/unmasking_model.h5')
print("Unmasking model saved.")

# 3. Train Emotion Classifier
print("\n--- Training Advanced Emotion Detection Model (ResNet) ---")
classifier = build_emotion_model()
classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# Callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss')
]

# Increase epochs for ResNet
classifier.fit(X_train_orig, y_train,
               epochs=20,
               batch_size=32,
               validation_data=(X_test_orig, y_test),
               callbacks=callbacks)

classifier.save('saved_models/emotion_model.h5')
print("Advanced emotion model saved.")

# 4. Optional: Fine-tune/Validate Pipeline
# We can check how well the classifier works on *reconstructed* images from the test set.
print("\n--- Validating Pipeline ---")
reconstructed_images = autoencoder.predict(X_test_masked)
loss, acc = classifier.evaluate(reconstructed_images, y_test)
print(f"Pipeline Accuracy (Masked -> Unmasked -> Detect): {acc*100:.2f}%")
