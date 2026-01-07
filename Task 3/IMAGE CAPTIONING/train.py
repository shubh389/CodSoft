import os
from typing import Dict, List
import numpy as np
from numpy.typing import NDArray


from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import load_img, img_to_array, pad_sequences, to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Add
from keras.models import Model, load_model


class SimpleTokenizer:
    """Minimal tokenizer compatible with Keras 3.
    Replaces deprecated tf.keras.preprocessing.text.Tokenizer.
    - Builds vocab from whitespace-separated tokens
    - Maps tokens -> indices (starting from 1; 0 reserved for padding)
    - Provides texts_to_sequences
    """

    def __init__(self) -> None:
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}

    def fit_on_texts(self, texts: List[str]) -> None:
        vocab: Dict[str, int] = {}
        for text in texts:
            for token in text.strip().split():
                if token:
                    vocab[token] = vocab.get(token, 0) + 1
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
                
            sequences.append(seq)
        return sequences



vgg = VGG16(weights="imagenet")

vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)



def extract_features(image_path: str) -> NDArray[np.float32] | NDArray[np.float64] | NDArray[np.int32]:
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = vgg.predict(image, verbose=0)
    return feature[0]


def load_captions(file_path: str) -> Dict[str, List[str]]:
    captions: Dict[str, List[str]] = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            image_id = parts[0].split('#')[0]
            caption = ' '.join(parts[1:])
            captions.setdefault(image_id, []).append(caption)
    return captions

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CAPTIONS_PATH = os.path.join(DATASET_DIR, "captions.txt")
IMAGES_DIR = os.path.join(DATASET_DIR, "image")  


def build_and_train() -> tuple[Model, SimpleTokenizer, int]:

    if not os.path.exists(CAPTIONS_PATH):
        print(f"Warning: Missing captions file: {CAPTIONS_PATH}. Skipping training.")
        raise FileNotFoundError("captions.txt not found")

    captions = load_captions(CAPTIONS_PATH)

    image_features: Dict[str, NDArray[np.float32] | NDArray[np.float64] | NDArray[np.int32]] = {}
    for img in captions.keys():
        path = os.path.join(IMAGES_DIR, img)
        if not os.path.exists(path):
            print(f"Warning: image not found, skipping: {path}")
            continue
        image_features[img] = extract_features(path)

    all_captions: List[str] = []
    for caps in captions.values():
        all_captions.extend(caps)

    tokenizer = SimpleTokenizer()
    tokenizer.fit_on_texts(all_captions if all_captions else ["startseq endseq"])

    vocab_size: int = len(tokenizer.word_index) + 1
    max_length: int = max((len(c.split()) for c in all_captions), default=2)

    x1_list: List[NDArray[np.float32] | NDArray[np.float64] | NDArray[np.int32]] = []
    x2_list: List[NDArray[np.int32]] = []
    y_list: List[NDArray[np.float32]] = []

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

    x1: NDArray[np.float32] | NDArray[np.float64] | NDArray[np.int32] = np.array(x1_list)
    x2: NDArray[np.int32] = np.array(x2_list)
    y: NDArray[np.float32] = np.array(y_list)

    # Build model
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

    if x1.size and x2.size and y.size:
        model.fit([x1, x2], y, epochs=20, batch_size=64)
        model.save(os.path.join(BASE_DIR, "image_caption_model.h5"))
        print("Training complete. Model saved to image_caption_model.h5")
    else:
        print("Warning: No training data available; model not trained.")

    return model, tokenizer, max_length


def load_or_build() -> tuple[Model, SimpleTokenizer, int]:
    model_path = os.path.join(BASE_DIR, "image_caption_model.h5")
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)
        
        if os.path.exists(CAPTIONS_PATH):
            caps = load_captions(CAPTIONS_PATH)
            all_caps: List[str] = []
            for c in caps.values():
                all_caps.extend(c)
            tokenizer = SimpleTokenizer()
            tokenizer.fit_on_texts(all_caps if all_caps else ["startseq endseq"])
            max_length = max((len(c.split()) for c in all_caps), default=2)
        else:
            tokenizer = SimpleTokenizer()
            tokenizer.fit_on_texts(["startseq endseq"])
            max_length = 2
        return model, tokenizer, max_length
    else:
        print("No saved model found; attempting to train.")
        return build_and_train()


def generate_caption(model: Model, tokenizer: SimpleTokenizer, max_length: int, photo: NDArray[np.float32] | NDArray[np.float64] | NDArray[np.int32]) -> str:
    text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([photo, seq], verbose=0)
        word_id = int(np.argmax(yhat))
        word = tokenizer.index_word.get(word_id)
        if word is None:
            break
        text += " " + word
        if word == "endseq":
            break
    return text


if __name__ == "__main__":
    try:
        model, tokenizer, max_length = load_or_build()
    except FileNotFoundError:
        print("Info: captions.txt missing; training skipped. If a saved model exists, it will be used for generation.")
        model_path = os.path.join(BASE_DIR, "image_caption_model.h5")
        if os.path.exists(model_path):
            model = load_model(model_path)
            tokenizer = SimpleTokenizer()
            tokenizer.fit_on_texts(["startseq endseq"])
            max_length = 2
        else:
            print("Info: No model available. Exiting.")
            raise SystemExit(0)

    test_image_path = os.path.join(BASE_DIR, "test.jpg")
    if os.path.exists(test_image_path):
        test_feature = extract_features(test_image_path)
        caption = generate_caption(model, tokenizer, max_length, np.expand_dims(test_feature, axis=0))
        print("Generated Caption:", caption)
    else:
        print(f"Info: Missing test image at {test_image_path}; add a test.jpg to generate a caption.")
