import json
import requests
import os
import firebase_admin
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from firebase_admin import credentials, db

# ------------------ Load secrets ------------------
with open("secrets.json", "r") as f:
    secrets = json.load(f)

GITHUB_TOKEN = secrets["GITHUB_TOKEN"]
GITHUB_USER = "Lu0SiN"
REPO_NAME = "sentweet-models"

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# ------------------ Versioning ------------------
def get_next_version(tag="v0.1"):
    base = tag.strip().lower().replace("v", "")
    parts = base.split(".")
    if len(parts) != 2:
        raise ValueError("Invalid version format.")
    major, minor = map(int, parts)
    minor += 1
    return f"v{major}.{minor}"

# Step 1: Get latest version from GitHub Releases
release_api = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/releases/latest"
latest = requests.get(release_api, headers=headers).json()
current_tag = latest.get("tag_name", "v0.1")
TAG = get_next_version(current_tag)
print(f"🔁 Latest version: {current_tag} → New version: {TAG}")


headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# ------------------ Firebase Init ------------------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://sentweetfeedback-default-rtdb.firebaseio.com/'
})

ref = db.reference("feedback")
data = ref.get()

if not data or len(data) < 500:
    print(f"❌ Not enough data to retrain. Found {len(data) if data else 0} entries.")
    exit()

df = pd.DataFrame.from_dict(data, orient='index')
texts = df['tweet'].astype(str).tolist()
labels = df['corrected'].astype('category')
y = labels.cat.codes.values
num_classes = len(set(y))

def get_next_version(tag="v0.1"):
    base = tag.strip().lower().replace("v", "")
    parts = base.split(".")
    if len(parts) != 2:
        raise ValueError("Invalid version format.")
    major, minor = map(int, parts)
    minor += 1
    return f"v{major}.{minor}"


# ------------------ Download from GitHub Release ------------------

def download_from_release(filename, tag="v0.1"):
    release_api = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/releases/tags/{tag}"
    release = requests.get(release_api, headers=headers).json()
    asset = next((a for a in release["assets"] if a["name"] == filename), None)
    if not asset:
        raise ValueError(f"{filename} not found in release {tag}")
    download_url = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/releases/assets/{asset['id']}"
    dl_headers = headers.copy()
    dl_headers["Accept"] = "application/octet-stream"
    r = requests.get(download_url, headers=dl_headers)
    with open(filename, "wb") as f:
        f.write(r.content)
    print(f"✅ Downloaded: {filename}")

download_from_release("word_index.json", tag="v0.1")
download_from_release("last_model.h5", tag="v0.1")

# ------------------ Update Tokenizer ------------------
with open("word_index.json", "r") as f:
    word_index = json.load(f)

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.word_index = word_index.copy()
tokenizer.fit_on_texts(texts)
updated_word_index = tokenizer.word_index

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=150)

# ------------------ Load/Train Model ------------------
try:
    model = load_model("last_model.h5")
    print("✅ Loaded previous model.")
except:
    print("❌ Could not load previous model.")
    model = Sequential([
        Embedding(input_dim=max(updated_word_index.values()) + 1, output_dim=64, input_length=150),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=5, batch_size=32, validation_split=0.1)
model.save("last_model.h5")

# ------------------ Export .tflite ------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()
with open("updated_model.tflite", "wb") as f:
    f.write(tflite_model)

# ------------------ Save Outputs ------------------
with open("word_index.json", "w") as f:
    json.dump(updated_word_index, f)

with open("model_version.txt", "w") as f:
    f.write(TAG)

print("✅ All files generated successfully.")

# ------------------ Upload to GitHub Release ------------------

# Create release
payload = {
    "tag_name": TAG,
    "name": f"Model Update {TAG}",
    "body": f"Auto-generated release from retraining script. ({len(df)} samples)",
    "draft": False,
    "prerelease": False
}

response = requests.post(f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/releases", headers=headers, json=payload)
res_data = response.json()
if "upload_url" not in res_data:
    print("❌ Failed to create release:", res_data)
    exit()

upload_url = res_data["upload_url"].split("{")[0]

def upload_to_release(file_path):
    file_name = os.path.basename(file_path)
    headers_upload = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Content-Type": "application/octet-stream"
    }
    with open(file_path, "rb") as f:
        upload_response = requests.post(
            f"{upload_url}?name={file_name}",
            headers=headers_upload,
            data=f
        )
    if upload_response.status_code == 201:
        print(f"✅ Uploaded {file_name}")
    else:
        print(f"❌ Upload failed for {file_name}:", upload_response.text)

for file in ["updated_model.tflite", "last_model.h5", "word_index.json", "model_version.txt"]:
    upload_to_release(file)

print("✅ Finished retraining and uploading.")
