import tensorflow as tf
from PIL import Image
import numpy as np
import textwrap
import matplotlib.pyplot as plt
import json

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from transformers import TFVisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

embedding_dim = 256
units = 512
vocab_size = 5001
max_length = 47
attention_features_shape = 64

# CNN Encoder
class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

# Attention Mechanism
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# RNN Decoder
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

# Load tokenizer
def load_tokenizer(tokenizer_path='tokenizer.json'):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer

# Image preprocessing
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

# Model to extract features
def build_feature_extraction_model():
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    return tf.keras.Model(new_input, hidden_layer)

# Evaluation function
def evaluate(image_path, encoder, decoder, tokenizer, feature_model):
    attention_plot = np.zeros((max_length, attention_features_shape))
    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image_path)[0], 0)
    img_tensor_val = feature_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result[:-1], attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    return result, attention_plot

# Attention plotting
def plot_attention(image_path, result, attention_plot):
    temp_image = np.array(Image.open(image_path))

    # Resize all attention maps to match the image and combine them
    attention_overlay = np.zeros((temp_image.shape[0], temp_image.shape[1]))

    for att in attention_plot:
        temp_att = np.resize(att, (8, 8))  # Assuming 8x8 attention
        temp_att = np.array(Image.fromarray(temp_att).resize(
            (temp_image.shape[1], temp_image.shape[0]),
            resample=Image.BILINEAR
        ))
        attention_overlay += temp_att

    # Normalize the combined attention map
    attention_overlay = attention_overlay / attention_overlay.max()

    # Plot the single image with combined attention overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(temp_image)
    plt.imshow(attention_overlay, cmap='jet', alpha=0.5)
    plt.title(" ".join(result), fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Hugging face implementation

# Load the TensorFlow-compatible model and tokenizer
model = TFVisionEncoderDecoderModel.from_pretrained("ydshieh/vit-gpt2-coco-en")
processor = ViTImageProcessor.from_pretrained("ydshieh/vit-gpt2-coco-en")
tokenizer = AutoTokenizer.from_pretrained("ydshieh/vit-gpt2-coco-en")

def caption_image_tf(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="tf").pixel_values

    # Generate caption
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def caption_and_display_image(image_path):
    # Generate caption
    caption = caption_image_tf(image_path)

    # Load and display the image
    image = Image.open(image_path).convert("RGB")
    wrapped_caption = "\n".join(textwrap.wrap(caption, width=60))

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(wrapped_caption, fontsize=14)
    plt.tight_layout()
    plt.show()
