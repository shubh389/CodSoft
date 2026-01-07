import os
import sys
from typing import Dict, List
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array, pad_sequences, to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Add
from tensorflow.keras.models import Model, load_model

# Minimal tokenizer compatible with Keras 3 (replaces deprecated tf.keras.preprocessing)
class SimpleTokenizer:
    def __init__(self) -> None:
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}

    def fit_on_texts(self, texts: List[str]) -> None:
        # Build vocabulary from whitespace-separated tokens
        vocab: Dict[str, int] = {}
        for text in texts:
            for token in text.strip().split():
                if token:
                    vocab[token] = vocab.get(token, 0) + 1
        # Assign indices starting from 1 (0 reserved for padding)
        # Keep deterministic ordering by frequency then alphabet
        sorted_tokens = sorted(vocab.items(), key=lambda x: (-x[1], x[0]))
        for idx, (tok, _) in enumerate(sorted_tokens, start=1):
            self.word_index[tok] = idx
            self.index_word[idx] = tok

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        sequences: List[List[int]] = []
        for text in texts:
            seq: List[int] = []
            for token in text.strip().split():
                idx = self.word_index.get(token)
                if idx is not None:
                    seq.append(idx)
                # silently drop OOV tokens
            sequences.append(seq)
        return sequences

# -------------------------------
# STEP 1: Load VGG16 CNN Model
# -------------------------------
vgg = VGG16(weights="imagenet")
vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)

# -------------------------------
# STEP 2: Extract Image Features
# -------------------------------
def extract_features(image_path: str) -> np.ndarray:
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = vgg.predict(image, verbose=0)
    return feature[0]

# -------------------------------
# STEP 3: Load Captions
# -------------------------------
def load_captions(file_path: str) -> Dict[str, List[str]]:
    captions: Dict[str, List[str]] = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            image_id = parts[0].split('#')[0]
            caption = ' '.join(parts[1:])
            captions.setdefault(image_id, []).append(caption)
    return captions

# Dataset paths
DATASET_DIR = "dataset"
CAPTIONS_PATH = os.path.join(DATASET_DIR, "captions.txt")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")

captions: Dict[str, List[str]] = {}
if os.path.exists(CAPTIONS_PATH):
    captions = load_captions(CAPTIONS_PATH)
else:
    print(f"Warning: Missing captions file: {CAPTIONS_PATH}. Skipping training.")


image_features: Dict[str, np.ndarray] = {}
if os.path.isdir(IMAGES_DIR) and captions:
    for img in captions.keys():
        path = os.path.join(IMAGES_DIR, img)
        image_features[img] = extract_features(path)


all_captions: List[str] = []
for caps in captions.values():
    all_captions.extend(caps)

tokenizer: SimpleTokenizer = SimpleTokenizer()
if all_captions:
    tokenizer.fit_on_texts(all_captions)
else:
    tokenizer.fit_on_texts(["startseq endseq"])

vocab_size: int = len(tokenizer.word_index) + 1
if all_captions:
    max_length: int = max(len(c.split()) for c in all_captions)
else:
    max_length: int = 2


x1_list: List[np.ndarray] = []
x2_list: List[np.ndarray] = []
y_list: List[np.ndarray] = []

for img, caps in captions.items():
    feature = image_features.get(img)
    if feature is None:
        continue
    for cap in caps:
        seq = tokenizer.texts_to_sequences([cap])[0]
        for i in range(1, len(seq)):
            in_seq = seq[:i]
            out_seq = seq[i]

            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical(out_seq, vocab_size)

            x1_list.append(feature)
            x2_list.append(in_seq)
            y_list.append(out_seq)

x1 = np.array(x1_list)
x2 = np.array(x2_list)
y = np.array(y_list)


image_input = Input(shape=(4096,))
image_dense = Dense(256, activation='relu')(image_input)

text_input = Input(shape=(max_length,))
text_embed = Embedding(vocab_size, 256)(text_input)
text_lstm = LSTM(256)(text_embed)


decoder_input = Add()([image_dense, text_lstm])
decoder = Dense(256, activation='relu')(decoder_input)
output = Dense(vocab_size, activation='softmax')(decoder)

model = Model(inputs=[image_input, text_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam')


trained = False
if x1.size and x2.size and y.size:
    model.fit([x1, x2], y, epochs=20, batch_size=64)
    model.save("image_caption_model.h5")
    trained = True
else:
    if os.path.exists("image_caption_model.h5"):
        model = load_model("image_caption_model.h5")
        trained = True
    else:
        print("Warning: No training data and no saved model found. Generation will be skipped.")


def generate_caption(photo: np.ndarray) -> str:
    text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([photo, seq], verbose=0)
        word_id = np.argmax(yhat)
        word = tokenizer.index_word.get(word_id)
        if word is None:
            break
        text += " " + word
        if word == "endseq":
            break
    return text


if trained and os.path.exists("test.jpg"):
    test_feature = extract_features("test.jpg")
    caption = generate_caption(np.expand_dims(test_feature, axis=0))
    print("Generated Caption:", caption)
else:
    if not trained:
        print("Info: Model not trained or loaded; skipping caption generation.")
    if not os.path.exists("test.jpg"):
        print("Info: Missing test.jpg in workspace; add an image to run generation.")
