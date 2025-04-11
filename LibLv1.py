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
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import shutil
import random
import winsound
import keyboard
import time
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras import  models


def play_beep(frequency=1000, duration=0.1):
    """
    Joue un bip sonore avec une fréquence et une durée spécifiées.

    Args:
        frequency (int): La fréquence du bip en Hertz. (par défaut 1000 Hz)
        duration (float): La durée du bip en secondes. (par défaut 0.1 seconde)
    """
    print("Le bip va commencer. Appuyez sur la touche Entrée pour arrêter...")

    # Écouter les événements de touches pendant que le bip est joué
    while True:
        winsound.Beep(frequency, int(duration * 1000))  
        time.sleep(duration)
        
        # Vérifier si la touche Entrée a été pressée
        if keyboard.is_pressed('enter'):  
            print("Bip arrêté.")
            break

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

# Vérification et correction des images
def preprocess_images(directory):
    valid_extensions = {".jpg", ".jpeg", ".png"}
    for class_name in os.listdir(directory):  
        class_path = os.path.join(directory, class_name)

        if not os.path.isdir(class_path):  
            continue

        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            file_ext = os.path.splitext(file_name)[-1].lower()

            if file_ext not in valid_extensions:
                print(f" Fichier ignoré (pas une image) : {file_path}")
                continue  

            try:
                with Image.open(file_path) as img:
                    img.verify()  
            except (IOError, SyntaxError) as e:
                print(f" Image corrompue supprimée : {file_path}")
                os.remove(file_path)  
                continue

def prepare_data(data_dir, target_size=(224, 224), batch_size=32, val_split=0.2):

    train_gen = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split = val_split,  
        subset="training",
        seed=42, 
        image_size=target_size,
        batch_size=batch_size
    )

# Jeu de test (20% des données)
    val_gen = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split = val_split,  
        subset = "validation",
        seed = 42,
        image_size = target_size,
        batch_size = batch_size
    )
    
    return train_gen, val_gen

def Create_Sequential(inputShape, ClassNb, useData_augmentation=True, ShowSummary=False, dropOut=0):
    model = keras.Sequential()
    
    if useData_augmentation:
        model.add(layers.RandomFlip('horizontal', input_shape=inputShape))
        model.add(layers.RandomRotation(0.2))
        model.add(layers.RandomZoom(0.2))
    
    model.add(layers.Rescaling(1./255, input_shape=inputShape))
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    if dropOut > 0 and dropOut <= 1:
        model.add(layers.Dropout(dropOut))

    model.add(layers.Flatten())

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(ClassNb, activation='softmax'))

    if ShowSummary:
        model.summary()

    return model

def create_mobilenetv2_model(input_shape, class_count, show_summary=False, dropout_rate=0.3, fine_tune_at=None):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'  # Utilise les poids pré-entraînés sur ImageNet
    )
    base_model.trainable = False  # Gèle les poids du modèle de base pour le moment

    inputs = keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)  # Prétraitement spécifique à MobileNetV2
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(class_count, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    if show_summary:
        model.summary()

    # Optionnel : fine-tuning de certaines couches si demandé
    if fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        print(f"Fine-tuning activé à partir de la couche {fine_tune_at}.")

    return model

def create_efficientnet_model(input_shape, class_count, show_summary=False, dropout_rate=0.3, fine_tune_at=None):
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # On gèle le modèle pour l'instant

    inputs = keras.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)  # Prétraitement spécifique
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(class_count, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    if show_summary:
        model.summary()

    # Optionnel : fine-tuning partiel
    if fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        print(f"Fine-tuning activé à partir de la couche {fine_tune_at}.")

    return model

def create_efficientnet_models(input_shape, class_count, version='B0', dropout_rate=0.3, fine_tune_at=None):
    """
    Crée un modèle EfficientNet spécifié par la version (B0 à B7).
    """
    # Dictionnaire des différentes versions de EfficientNet
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
        raise ValueError("Version de EfficientNet non valide. Choisissez parmi ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'].")
    
    base_model = efficientnet_versions[version](
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Geler les poids du modèle pré-entraîné
    
    # Création du modèle
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)  # Prétraitement des images
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(class_count, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    # Fine-tuning si nécessaire
    if fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        print(f"Fine-tuning activé à partir de la couche {fine_tune_at}.")
    
    return model

def load_model_from_weights_and_compile(model, weights_file):
    """
    Charge les poids d'un modèle et compile ce dernier si le modèle existe.
    
    Args:
        model (tf.keras.Model): Le modèle à recharger.
        weights_file (str): Le chemin vers le fichier des poids.
    
    Returns:
        model (tf.keras.Model): Le modèle avec les poids chargés et compilé (si le modèle existe).
        si erreur il retourne none
    """
    if model is not None:  
        try:
            model.load_weights(weights_file)
            print(f"Poids chargés avec succès depuis {weights_file}")
        except Exception as e:
            print(f"Erreur lors du chargement des poids : {e}")
            return None

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        print("Modèle compilé avec succès.")
        return model
    else:
        print("Le modèle n'a pas été créé ou est invalide.")
        return None

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

def train_and_test_efficientnets(input_shape, class_count, train_ds, val_ds, start_From = 'B0',nb_epochs=10, dropout_rate=0.3  , fine_tune_at = None ,NameDtaset = "Bin" ):
    """
    Entraîne et teste EfficientNet de B0 à B7.
    """
    results = {}
    found = False
    for version in ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']:
        print(f"\n--- Entrainement de EfficientNet-{version} ---")
        if(not found and version != start_From):
            print(f"skipping the {version} model")
            continue
        found = True
        model = create_efficientnet_models(input_shape, class_count, version, dropout_rate  , fine_tune_at)
        history = TrainModel(model, train_ds, val_ds, nb_epochs ,modelCheckpoint=f"models/EfficientNet{version}_{NameDtaset}_best_weights.h5")
        # Évaluation sur le jeu de validation
        loss, accuracy = model.evaluate(val_ds)
        # Stocker les résultats
        results[version] = {
            'history': history ,
            'loss' : loss , 
            'accuracy'  : accuracy
        }
        
        # Affichage des courbes de performance pour chaque modèle
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Accuracy Entrainement')
        plt.plot(history.history['val_accuracy'], label='Accuracy Validation')
        plt.title(f'Accuracy de EfficientNet-{version}')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Loss Entrainement')
        plt.plot(history.history['val_loss'], label='Loss Validation')
        plt.title(f'Loss de EfficientNet-{version}')
        plt.legend()

        plt.show()
        plot_confusion_matrix(model ,val_ds , list(val_ds.class_names) ,title = f"Matrice de Confusion-{version}-")
    
    return results

def displayHistoryData(acc, val_acc, loss, val_loss, nbepochs):
    plt.figure(figsize=(16, 8))
    epochs_range = range(nbepochs)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

def test_model(model, test_set=None, image_path=None, class_names=None, image_size=(224, 224)):
    """
    Teste un modèle soit sur un jeu de données complet (test_set),
    soit sur une seule image en donnant son chemin (image_path).

    Args:
        model: Le modèle entraîné.
        test_set: tf.data.Dataset, le dataset de test.
        image_path: str, chemin vers une image individuelle.
        class_names: list, noms des classes.
        image_size: tuple, taille (h, w) pour redimensionner l'image.

    Returns:
        None
    """
    if test_set is not None:
        # Évaluation sur l'ensemble du test
        loss, accuracy = model.evaluate(test_set)
        print(f"Évaluation sur le jeu de test - Accuracy: {accuracy:.2f}, Loss: {loss:.4f}")
    
    if image_path is not None:
        # Prédiction sur une image unique
        img = image.load_img(image_path, target_size=image_size)
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Ajoute la dimension batch

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        print(f"Classe prédite: {class_names[predicted_class]} avec une confiance de {confidence:.2f}")

        plt.imshow(img)
        plt.title(f"Prédit: {class_names[predicted_class]} ({confidence:.2f})")
        plt.axis('off')
        plt.show()

def plot_confusion_matrix(model, test_set, class_names, title="Matrice de Confusion"):
    """
    Affiche la matrice de confusion normalisée (en pourcentage) pour un modèle donné.

    Args:
        model: Le modèle entraîné.
        test_set: Dataset TensorFlow contenant des (images, labels).
        class_names: Liste des noms des classes.
        title: Titre du graphique (par défaut "Matrice de Confusion").

    Returns:
        None
    """
    y_true = []
    y_pred = []

    # Itérer sur le dataset pour obtenir les vraies étiquettes et les prédictions
    for images, labels in test_set:
        y_true.extend(labels.numpy())
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)

    # Normaliser en pourcentage
    cm_percentage = (cm.astype('float') / cm.sum(axis=1, keepdims=True)) * 100

    # Afficher la somme de chaque ligne (devrait être proche de 100)
    print("Vérification des lignes (%):", cm_percentage.sum(axis=1))

    # Affichage du heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    plt.show()

def create_balanced_no_photos_folder(base_path=".\\dataset_binary"):
    photo_folder = os.path.join(base_path, "Photo")
    no_photos_folder = os.path.join(base_path, "No_photos")
    other_folders = [
        os.path.join(base_path, "Painting"),
        os.path.join(base_path, "Schematics"),
        os.path.join(base_path, "Sketch"),
        os.path.join(base_path, "Text")
    ]
    # Supprimer le dossier No_photos s’il existe et le recréer
    if os.path.exists(no_photos_folder):
        shutil.rmtree(no_photos_folder)
    os.makedirs(no_photos_folder)

    # Vérifie si un fichier est une image corrompue
    def is_file_corrupted(filepath):
        try:
            with Image.open(filepath) as img:
                img.verify()
            return False
        except:
            return True

    # Récupérer les fichiers valides dans les autres dossiers
    valid_files = {}
    for folder in other_folders:
        if os.path.exists(folder):
            all_files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))
            ]
            valid = [f for f in all_files if not is_file_corrupted(f)]
            valid_files[folder] = valid

    # Compter les images dans le dossier "Photo"
    photo_files = [
        f for f in os.listdir(photo_folder)
        if os.path.isfile(os.path.join(photo_folder, f))
    ]
    n_photo_files = len(photo_files)

    # Calcul du total d'images valides et copie proportionnelle
    total_valid = sum(len(files) for files in valid_files.values())

    for folder, files in valid_files.items():
        ratio = len(files) / total_valid
        n_to_copy = round(ratio * n_photo_files)
        selected_files = random.sample(files, min(n_to_copy, len(files)))

        for file_path in selected_files:
            shutil.copy(file_path, no_photos_folder)

    print("No_photos folder created with matching proportions and cleaned of corrupted files.")

def show_misclassified_images(model, dataset, class_names, image_size=(224, 224), max_images=10, num_columns=5):
    """
    Affiche les images mal classées à partir d'un dataset de validation ou test,
    sous forme d'une matrice spécifiée par le nombre de colonnes.

    Args:
        model: Le modèle entraîné.
        dataset: Dataset TensorFlow (ex: val_ds/test_ds).
        class_names: Liste des noms des classes.
        image_size: Tuple pour redimensionner les images.
        max_images: Nombre maximum d'images à afficher.
        num_columns: Nombre de colonnes dans la grille d'affichage des images.

    """
    misclassified = []

    for batch_images, batch_labels in dataset:
        preds = model.predict(batch_images)
        predicted_classes = tf.argmax(preds, axis=1)
        true_classes = tf.cast(batch_labels, tf.int64)

        # Comparaison
        for i in range(len(batch_images)):
            if predicted_classes[i] != true_classes[i]:
                misclassified.append((batch_images[i], predicted_classes[i], true_classes[i]))
            if len(misclassified) >= max_images:
                break
        if len(misclassified) >= max_images:
            break

    # Calcul du nombre de lignes nécessaires
    num_rows = (len(misclassified) + num_columns - 1) // num_columns  # Calcul pour le nombre de lignes

    # Affichage
    plt.figure(figsize=(15, num_rows * 3))  # Ajuste la hauteur de la figure en fonction du nombre de lignes
    for idx, (img, pred, true) in enumerate(misclassified):
        plt.subplot(num_rows, num_columns, idx + 1)
        plt.imshow(img.numpy().astype("uint8"))
        plt.title(f"Prédit: {class_names[pred]}\nRéel: {class_names[true]}", fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
