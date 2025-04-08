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
            # Activer la gestion dynamique de la mémoire
            tf.config.experimental.set_memory_growth(physical_devices[min_gpu_index], True)
            
            # Restreindre TensorFlow à utiliser uniquement le GPU spécifié
            tf.config.set_visible_devices(physical_devices[min_gpu_index], 'GPU')
            
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
