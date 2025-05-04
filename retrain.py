# ✅ GPU Check
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name:
    print(f"✅ GPU detected: {device_name}")
else:
    print("❌ GPU not detected, training will be slower")

# ✅ Imports
import os, json, requests, re
import pandas as pd
import numpy as np
from collections import OrderedDict, Counter
import firebase_admin
from firebase_admin import credentials, db
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Input

# %%
# ✅ Utility: Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

# ✅ Load secrets
with open("secrets.json", "r") as f:
    secrets = json.load(f)

GITHUB_TOKEN = secrets["GITHUB_TOKEN"]
GITHUB_USER = "Lu0SiN"
REPO_NAME = "sentweet-models"

# ✅ Write Firebase key
with open("serviceAccountKey.json", "w") as f:
    json.dump({k: v for k, v in secrets.items() if k != "GITHUB_TOKEN"}, f)

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# %%
# ✅ Get next version tag
def get_next_version(tag="v0.1"):
    base = tag.strip().lower().replace("v", "")
    major, minor = map(int, base.split("."))
    return f"v{major}.{minor + 1}"

release_api = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/releases/latest"
latest = requests.get(release_api, headers=headers).json()
current_tag = latest.get("tag_name", "v0.1")
TAG = get_next_version(current_tag)
print(f"🔁 Latest version: {current_tag} → New version: {TAG}")

# %%
# ✅ Firebase Init
cred = credentials.Certificate("serviceAccountKey.json")
try:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://sentweetfeedback-default-rtdb.firebaseio.com/'
    })
except ValueError:
    pass

ref = db.reference("feedback")
data = ref.get()
firebase_df = pd.DataFrame.from_dict(data, orient="index") if data else pd.DataFrame()
firebase_df = firebase_df[["corrected", "tweet"]].dropna()

# ✅ Feedback count threshold logic
LAST_COUNT_FILE = "last_feedback_count.txt"

def get_last_feedback_count():
    if os.path.exists(LAST_COUNT_FILE):
        with open(LAST_COUNT_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_feedback_count(count):
    with open(LAST_COUNT_FILE, "w") as f:
        f.write(str(count))

current_feedback_count = len(firebase_df)
last_feedback_count = get_last_feedback_count()

if current_feedback_count - last_feedback_count < 200:
    print(f"⛔ Not enough new feedback samples. Only {current_feedback_count - last_feedback_count} new.")
    exit(0)
else:
    print(f"✅ Enough new feedback samples: {current_feedback_count - last_feedback_count}. Proceeding...")

# %%
# ✅ Load CSV Datasets
kaggle_df = pd.read_csv("kaggle_dataset.csv")[["SENTIMENT", "TWEET"]].dropna()
kaggle_df.columns = ["corrected", "tweet"]

sent140_df = pd.read_csv("sentiment140_labeled.csv")[["SENTIMENT", "TWEET"]].dropna()
sent140_df.columns = ["corrected", "tweet"]

print(f"✅ Feedback: {len(firebase_df)}, Kaggle: {len(kaggle_df)}, Sent140: {len(sent140_df)}")

# ✅ Merge and prepare labels
merged_df = pd.concat([kaggle_df, sent140_df, firebase_df], ignore_index=True)
texts = merged_df["tweet"].astype(str).apply(clean_text).tolist()
labels = merged_df["corrected"].astype("category")
labels = labels.cat.set_categories(["Negative", "Neutral", "Positive", "Irrelevant"])
y = labels.cat.codes.values
num_classes = len(set(y))

# ✅ Download and update word index
def download_from_release(filename, tag="v0.1"):
    rel = requests.get(
        f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/releases/tags/{tag}",
        headers=headers
    ).json()
    asset = next((a for a in rel.get("assets", []) if a["name"] == filename), None)
    if not asset:
        raise ValueError(f"{filename} not found in release {tag}")
    url = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/releases/assets/{asset['id']}"
    r = requests.get(url, headers={**headers, "Accept": "application/octet-stream"})
    with open(filename, "wb") as f:
        f.write(r.content)
    print(f"✅ Downloaded: {filename}")

download_from_release("word_index.json", current_tag)

try:
    with open('word_index.json', 'r') as f:
        old_word_index = json.load(f)
except FileNotFoundError:
    old_word_index = {}

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
new_word_index = tokenizer.word_index.copy()
max_index = max(new_word_index.values(), default=0)

for word, index in old_word_index.items():
    if word not in new_word_index:
        max_index += 1
        new_word_index[word] = max_index

merged_word_index = OrderedDict(sorted(new_word_index.items(), key=lambda x: x[1]))
filtered_word_index = {word: idx for word, idx in merged_word_index.items() if idx < 30000}
vocab_size = len(filtered_word_index) + 2

# Save updated index
with open("word_index.json", "w") as f:
    json.dump(merged_word_index, f)
with open("vocab_size.txt", "w") as f:
    f.write(str(vocab_size))

# Tokenize
tokenizer.word_index = filtered_word_index
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=150)

# Save tokenizer config
with open("tokenizer_config.json", "w") as f:
    f.write(tokenizer.to_json())

# %%
# ✅ Build and train model
model = Sequential([
    Input(shape=(150,)),
    Embedding(input_dim=vocab_size, output_dim=128),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
    run_eagerly=True
)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('last_model.h5', save_best_only=True, monitor='val_loss')

history = model.fit(
    X, y,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop, model_checkpoint],
    verbose=1
)

# ✅ Save feedback count
save_feedback_count(current_feedback_count)

# %%
# ✅ TFLite export
model = load_model("last_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open("updated_model.tflite", "wb") as f:
    f.write(tflite_model)

# ✅ Test
interpreter = tf.lite.Interpreter(model_path="updated_model.tflite")
interpreter.allocate_tensors()
print("✅ TFLite model validated.")

# %%
# ✅ Save version
with open("model_version.txt", "w") as f:
    f.write(TAG)

# ✅ Upload release to GitHub
payload = {
    "tag_name": TAG,
    "name": f"Model Update {TAG}",
    "body": f"Auto-generated release with {len(merged_df)} samples",
    "draft": False,
    "prerelease": False
}
res = requests.post(f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/releases", headers=headers, json=payload).json()
upload_url = res.get("upload_url", "").split("{")[0]

def upload_to_release(file_path):
    file_name = os.path.basename(file_path)
    headers_upload = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Content-Type": "application/octet-stream"
    }
    with open(file_path, "rb") as f:
        resp = requests.post(f"{upload_url}?name={file_name}", headers=headers_upload, data=f)
        if resp.status_code == 201:
            print(f"✅ Uploaded: {file_name}")
        else:
            print(f"❌ Upload failed for {file_name}: {resp.text}")

for file in ["updated_model.tflite", "last_model.h5", "word_index.json", "model_version.txt"]:
    upload_to_release(file)

print("✅ Retraining complete and uploaded.")

