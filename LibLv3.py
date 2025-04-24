import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from PIL import Image
import matplotlib.pyplot as plt

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)
    return tokenizer_from_json(tokenizer_json)


def decode_caption(predicted_ids, tokenizer):
    words = [tokenizer.index_word.get(i, '') for i in predicted_ids]
    caption = []
    for word in words:
        if word == '<end>':
            break
        if word not in ['<start>', '<pad>', '']:
            caption.append(word)
    return ' '.join(caption)


def evaluate(image_path, encoder, decoder, tokenizer, max_length, attention_features_shape, image_features_extract_model):
    # Load and preprocess image
    temp_input = tf.expand_dims(load_image(image_path), 0)
    img_tensor_val = image_features_extract_model(temp_input)

    # Dynamically reshape features from (batch, H, W, C) -> (batch, H*W, C)
    img_tensor_val = tf.reshape(img_tensor_val, (tf.shape(img_tensor_val)[0], -1, tf.shape(img_tensor_val)[-1]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[i] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(predicted_id)

        if tokenizer.index_word.get(predicted_id, "") == '<end>':
            break

        dec_input = tf.expand_dims([predicted_id], 0)

    return decode_caption(result, tokenizer), attention_plot



def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def plot_attention(image_path, result, attention_plot):
    image = Image.open(image_path)
    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
        ax.axis('off')

    plt.tight_layout()
    plt.show()

