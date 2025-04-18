from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
import shutil
import random
import winsound
import keyboard
import time
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras import  models

import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim



def is_gpu_available(min_gpu_index=0):
    """
    Vérifie si un GPU est disponible et configure la mémoire dynamique pour celui-ci.
    
    Args:
        min_gpu_index (int): L'index minimal du GPU à utiliser (par défaut 0).
    
    Returns:
        bool: True si le GPU est disponible et configuré, False sinon.
    """
    print("TensorFlow version:", tf.__version__)
    
    physical_devices = tf.config.list_physical_devices('GPU')
    print("GPU disponibles:", physical_devices)
    
    if len(physical_devices) > min_gpu_index:
        try:
            # Restreindre TensorFlow à utiliser uniquement le GPU spécifié
            tf.config.set_visible_devices(physical_devices[min_gpu_index], 'GPU')
            # Activer la gestion dynamique de la mémoire
            tf.config.experimental.set_memory_growth(physical_devices[min_gpu_index], True)
            print(f"GPU {min_gpu_index} est activé avec gestion dynamique de la mémoire.")
            return True
        except Exception as e:
            print(f" Erreur lors de la configuration du GPU : {e}")
            return False
    else:
        print("Pas assez de GPU disponibles.")
        return False

# ----------- Préparation des données -----------
def prepare_data(data_dir, target_size=(224, 224), batch_size=32, val_split=0.2):
    train_gen = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=42,
        image_size=target_size,
        batch_size=batch_size
    )

    val_gen = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=42,
        image_size=target_size,
        batch_size=batch_size
    )


    return train_gen, val_gen
def prepare_autoencoder_dataset(dataset, noise=True, noise_factor=0.3):
    def map_fn(x, y):
        x = tf.cast(x, tf.float32) / 255.0 if tf.reduce_max(x) > 1.0 else x
        if noise:
            x_noisy = add_noise(x, noise_factor)
            return x_noisy, x
        else:
            return x, x
    return dataset.map(map_fn)
# ----------- Ajout de bruit -----------
def add_noise(images, noise_factor=0.3):
    # S'assurer que les images sont float32 et dans [0,1]
    images = tf.cast(images, tf.float32) / 255.0 if tf.reduce_max(images) > 1.0 else images
    noise = noise_factor * tf.random.normal(shape=tf.shape(images))
    noisy_images = images + noise
    return tf.clip_by_value(noisy_images, 0.0, 1.0)

import matplotlib.pyplot as plt

def display_dataset_samples(dataset, n=5):
    """
    Affiche les premières images bruitées et originales du dataset autoencoder.
    :param dataset: Dataset prétraité avec (x_noisy, x_clean)
    :param n: Nombre d'exemples à afficher
    """
    for noisy_imgs, clean_imgs in dataset.take(1):  # On prend juste 1 batch
        plt.figure(figsize=(12, 4))
        for i in range(n):
            # Image bruitée
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(noisy_imgs[i].numpy())
            plt.title("Bruitée")
            plt.axis("off")

            # Image propre (cible)
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(clean_imgs[i].numpy())
            plt.title("Originale")
            plt.axis("off")
        plt.tight_layout()
        plt.show()


def create_autoencoder(input_shape):
    input_img = keras.Input(shape=input_shape)

    # Encodeur (un peu plus profond)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(input_img)  # 224 → 112
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(x)          # 112 → 56
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(x)          # 56 → 28
    encoded = x  # (28, 28, 16)

    # Décodeur
    x = layers.UpSampling2D((2, 2))(encoded)     # 28 → 56
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)           # 56 → 112
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)           # 112 → 224
    decoded = layers.Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    autoencoder.summary()
    return autoencoder


# ----------- Construction de l'autoencodeur -----------
def create_combined_autoencoder_efficientnet(input_shape, class_count, autoencoder, version='B0', dropout_rate=0.3, fine_tune_at=None):
    # On fige l'autoencodeur pour ne pas le réentraîner
    autoencoder.trainable = False

    # Choisir la version de EfficientNet
    efficientnet_versions = {
        'B0': EfficientNetB0,
        'B1': EfficientNetB1,
        'B2': EfficientNetB2,
        'B3': EfficientNetB3,
        'B4': EfficientNetB4,
        'B5': EfficientNetB5,
        'B6': EfficientNetB6,
        'B7': EfficientNetB7
    }
    if version not in efficientnet_versions:
        raise ValueError(f"Version {version} non supportée.")
    
    base_model = efficientnet_versions[version](
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    
    # Pipeline combiné : Autoencodeur → EfficientNet
    x = autoencoder(inputs)  # Denoising / encoding
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(class_count, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    if fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        print(f"Fine-tuning activé à partir de la couche {fine_tune_at}.")
    
    return model

# ----------- Générateur de bruit -----------
def noisy_dataset(dataset):
    return dataset.map(lambda x, y: (add_noise(x), y))

def display_comparison(autoencoder, dataset, n=5, noisy=True):
    for batch in dataset.take(1):
        images, _ = batch  # On ignore les labels
        images = tf.cast(images, tf.float32) / 255.0
        noisy_images = add_noise(images) if noisy else images
        reconstructed = autoencoder.predict(noisy_images)

        plt.figure(figsize=(20, 6))
        for i in range(n):
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(noisy_images[i])
            plt.title("Bruitée" if noisy else "Originale")
            plt.axis("off")

            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(reconstructed[i])
            plt.title("Reconstruite")
            plt.axis("off")

            ax = plt.subplot(3, n, i + 1 + 2*n)
            plt.imshow(images[i])
            plt.title("Propre")
            plt.axis("off")
        plt.show()



def TrainModel(model, train_set, test_set, nbEpochs=10, endEpochCallback=None, UseEarlyStopping=True, modelCheckpoint=None):
    callbacks = []
    if modelCheckpoint and os.path.exists(modelCheckpoint):
        try:
            model.load_weights(modelCheckpoint)
            print(f"Poids chargés depuis '{modelCheckpoint}'.")
        except Exception as e:
            print(f"Erreur lors du chargement des poids : {e}")

    if endEpochCallback:
        callbacks.append(endEpochCallback)

    if UseEarlyStopping:
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        callbacks.append(early_stopping)

    if modelCheckpoint:
        checkpoint = ModelCheckpoint(modelCheckpoint, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
        callbacks.append(checkpoint)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_set,
        validation_data=test_set,
        epochs=nbEpochs,
        callbacks=callbacks
    )

    

    return history


def evaluate_autoencoder(autoencoder, dataset, n_samples=100):
    mse_scores = []
    mae_scores = []
    ssim_scores = []

    sample_count = 0

    for noisy_imgs, clean_imgs in dataset:
        preds = autoencoder.predict(noisy_imgs, verbose=0)

        for i in range(len(noisy_imgs)):
            true_img = clean_imgs[i].numpy()
            pred_img = preds[i]

            # MSE & MAE
            mse_scores.append(mean_squared_error(true_img.flatten(), pred_img.flatten()))
            mae_scores.append(mean_absolute_error(true_img.flatten(), pred_img.flatten()))

            # SSIM (convert to grayscale first)
            ssim_score = ssim(
                tf.image.rgb_to_grayscale(true_img).numpy().squeeze(),
                tf.image.rgb_to_grayscale(pred_img).numpy().squeeze(),
                data_range=1.0
            )
            ssim_scores.append(ssim_score)

            sample_count += 1
            if sample_count >= n_samples:
                break
        if sample_count >= n_samples:
            break

    # Résumé
    print(f"Autoencoder Evaluation on {sample_count} samples:")
    print(f" - Mean MSE  : {np.mean(mse_scores):.4f}")
    print(f" - Mean MAE  : {np.mean(mae_scores):.4f}")
    print(f" - Mean SSIM : {np.mean(ssim_scores):.4f} (1 = perfect match)")

    return {
        'mse': mse_scores,
        'mae': mae_scores,
        'ssim': ssim_scores
    }


def plot_autoencoder_metrics(metrics):
    mse = metrics['mse']
    mae = metrics['mae']
    ssim = metrics['ssim']
    x = list(range(len(mse)))  # index des images

    plt.figure(figsize=(18, 5))

    # MSE
    plt.subplot(1, 3, 1)
    plt.plot(x, mse, label="MSE", color='royalblue')
    plt.xlabel("Image Index")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE par image")
    plt.grid(True)

    # MAE
    plt.subplot(1, 3, 2)
    plt.plot(x, mae, label="MAE", color='darkorange')
    plt.xlabel("Image Index")
    plt.ylabel("Mean Absolute Error")
    plt.title("MAE par image")
    plt.grid(True)

    # SSIM
    plt.subplot(1, 3, 3)
    plt.plot(x, ssim, label="SSIM", color='green')
    plt.xlabel("Image Index")
    plt.ylabel("SSIM (1 = meilleur)")
    plt.title("SSIM par image")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
