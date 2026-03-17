import tensorflow as tf
from tensorflow.keras import layers, models

def build_unmasking_model(input_shape=(64, 64, 1)):
    """
     Convolutional Autoencoder for unmasking/inpainting.
    """
    input_img = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = models.Model(input_img, decoded)
    return model

def build_resnet_emotion_model(input_shape=(64, 64, 1), num_classes=7):
    """
    Advanced ResNet-based model for higher precision emotion detection.
    Uses Transfer Learning principles with a custom head.
    """
    # ResNet50V2 expects RGB (3 channels). We expand our grayscale to 3 channels.
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(input_shape[0], input_shape[1], 3)
    )
    
    # Freeze the base model layers (start with transfer learning)
    base_model.trainable = False
    
    # Create the model
    inputs = layers.Input(shape=input_shape)
    
    # Convert Grayscale (1 channel) to RGB-like (3 channels)
    x = layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x))(inputs)
    
    # Pass through base ResNet
    x = base_model(x, training=False)
    
    # Custom Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

def build_emotion_model(input_shape=(64, 64, 1), num_classes=7):
    """
    Wrapper for choosing the model. Defaulting to ResNet now for high accuracy.
    """
    return build_resnet_emotion_model(input_shape, num_classes)
