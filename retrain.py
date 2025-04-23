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
import matplotlib.pyplot as plt
from collections import OrderedDict, Counter
import firebase_admin
from firebase_admin import credentials, db
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ✅ Text Cleaner
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

# ✅ Load secrets.json
with open("secrets.json", "r") as f:
    secrets = json.load(f)

GITHUB_TOKEN = secrets["GITHUB_TOKEN"]
GITHUB_USER = "Lu0SiN"
REPO_NAME = "sentweet-models"

# ✅ Write Firebase Key
with open("serviceAccountKey.json", "w") as f:
    json.dump({k: v for k, v in secrets.items() if k != "GITHUB_TOKEN"}, f)

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# ✅ Get Next Version
def get_next_version(tag="v0.1"):
    base = tag.strip().lower().replace("v", "")
    major, minor = map(int, base.split("."))
    return f"v{major}.{minor + 1}"

release_api = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/releases/latest"
latest = requests.get(release_api, headers=headers).json()
current_tag = latest.get("tag_name", "v0.1")
TAG = get_next_version(current_tag)
print(f"🔁 Latest version: {current_tag} → New version: {TAG}")

# ✅ Firebase Init
cred = credentials.Certificate("serviceAccountKey.json")
try:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://sentweetfeedback-default-rtdb.firebaseio.com/'
    })
except ValueError:
    pass

# ✅ Load Firebase Feedback
ref = db.reference("feedback")
data = ref.get()
firebase_df = pd.DataFrame.from_dict(data, orient="index") if data else pd.DataFrame()
firebase_df = firebase_df[["corrected", "tweet"]].dropna()

# ✅ Load Datasets
kaggle_df = pd.read_csv("kaggle_dataset.csv")[["SENTIMENT", "TWEET"]].dropna()
kaggle_df.columns = ["corrected", "tweet"]

sent140_df = pd.read_csv("sentiment140_balanced.csv")[["SENTIMENT", "TWEET"]].dropna()
sent140_df.columns = ["corrected", "tweet"]

print(f"✅ Feedback: {len(firebase_df)}, Kaggle: {len(kaggle_df)}, Sent140: {len(sent140_df)}")

# ✅ Merge and Prepare
merged_df = pd.concat([firebase_df, kaggle_df, sent140_df], ignore_index=True)
texts = merged_df["tweet"].astype(str).apply(clean_text).tolist()
labels = merged_df["corrected"].astype("category")
labels = labels.cat.set_categories(["Negative", "Neutral", "Positive", "Irrelevant"])
y = labels.cat.codes.values

# ✅ Class Distribution Plot
class_distribution = Counter(y)
label_map = dict(enumerate(labels.cat.categories))
class_distribution_named = {label_map[k]: v for k, v in class_distribution.items()}

plt.figure(figsize=(8, 4))
plt.bar(class_distribution_named.keys(), class_distribution_named.values(), color="skyblue")
plt.title("✅ Class Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# ✅ Download Previous Word Index
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

# ✅ Tokenizer Update (Limit to top 10k words)
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
new_word_index = tokenizer.word_index.copy()
for word, index in old_word_index.items():
    if word not in new_word_index:
        new_word_index[word] = len(new_word_index) + 1

# Filter and apply
filtered_word_index = {w: i for w, i in new_word_index.items() if i < 10000}
tokenizer.word_index = filtered_word_index
vocab_size = len(filtered_word_index) + 2

with open("word_index.json", "w") as f:
    json.dump(filtered_word_index, f)
with open("vocab_size.txt", "w") as f:
    f.write(str(vocab_size))

# ✅ Preprocess Inputs
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=150)

# ✅ Build Model
embedding_dim = 64
lstm_units = 32
dropout_rate = 0.5

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=150),
    Bidirectional(LSTM(lstm_units, return_sequences=True)),
    Dropout(dropout_rate),
    Bidirectional(LSTM(lstm_units)),
    Dropout(dropout_rate),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

class_weights = {i: (1.0 / labels.value_counts().iloc[i]) for i in labels.cat.codes.unique()}

history = model.fit(
    X, y,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop, model_checkpoint, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# ✅ Save Light Inference Model
model.save("inference_model.h5", include_optimizer=False)

# ✅ Export to Quantized TFLite
model = tf.keras.models.load_model("inference_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()
with open("updated_model.tflite", "wb") as f:
    f.write(tflite_model)

# ✅ Verify
interpreter = tf.lite.Interpreter(model_path="updated_model.tflite")
interpreter.allocate_tensors()
print("✅ TFLite model ready for Android.")

# ✅ Save Version Tag
with open("model_version.txt", "w") as f:
    f.write(TAG)

# ✅ Upload to GitHub
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
        print(f"✅ Uploaded: {file_name}" if resp.status_code == 201 else f"❌ Failed: {file_name} - {resp.text}")

for file in ["updated_model.tflite", "inference_model.h5", "word_index.json", "model_version.txt"]:
    upload_to_release(file)

print("✅ Training, export and upload complete.")
