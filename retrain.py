import json, os, requests
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from firebase_admin import credentials, db
import firebase_admin

# ✅ True Eager Execution (compatible with TF 2.x and GitHub Actions)
tf.compat.v1.disable_eager_execution()  # First make sure everything is clean
tf.compat.v1.enable_eager_execution()
print("✅ Eager execution force-enabled.")

# ------------------ Load secrets ------------------
with open("secrets.json", "r") as f:
    secrets = json.load(f)

GITHUB_TOKEN = secrets["GITHUB_TOKEN"]
GITHUB_USER = "Lu0SiN"
REPO_NAME = "sentweet-models"

# 🔐 Extract and write Firebase service account key
with open("serviceAccountKey.json", "w") as f:
    json.dump({k: v for k, v in secrets.items() if k != "GITHUB_TOKEN"}, f)

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# ------------------ Versioning ------------------
def get_next_version(tag="v0.1"):
    base = tag.strip().lower().replace("v", "")
    major, minor = map(int, base.split("."))
    return f"v{major}.{minor + 1}"

release_api = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/releases/latest"
latest = requests.get(release_api, headers=headers).json()
current_tag = latest.get("tag_name", "v0.1")
TAG = get_next_version(current_tag)
print(f"🔁 Latest version: {current_tag} → New version: {TAG}")

# ------------------ Firebase Init ------------------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://sentweetfeedback-default-rtdb.firebaseio.com/'
})

ref = db.reference("feedback")
data = ref.get()
firebase_df = pd.DataFrame.from_dict(data, orient="index") if data else pd.DataFrame()

# ------------------ Load Kaggle Dataset ------------------
kaggle_df = pd.read_csv("kaggle_dataset.csv")
kaggle_df = kaggle_df[["SENTIMENT", "TWEET"]].dropna()
kaggle_df.columns = ["corrected", "tweet"]

# ------------------ Merge both datasets ------------------
#if firebase_df.empty or len(firebase_df) < 200:
  #  print(f"❌ Not enough feedback data to retrain. Found {len(firebase_df)} entries.")
 #   exit()

print(f"ℹ️ Feedback samples found: {len(firebase_df)}. Proceeding with training anyway.")

firebase_df = firebase_df[["corrected", "tweet"]].dropna()
merged_df = pd.concat([firebase_df, kaggle_df], ignore_index=True)

texts = merged_df["tweet"].astype(str).tolist()
labels = merged_df["corrected"].astype("category")
y = labels.cat.codes.values
num_classes = len(set(y))

# ------------------ Download previous model + tokenizer ------------------
def download_from_release(filename, tag="v0.1"):
    rel = requests.get(
        f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/releases/tags/{tag}",
        headers=headers
    ).json()
    asset = next((a for a in rel["assets"] if a["name"] == filename), None)
    if not asset:
        raise ValueError(f"{filename} not found in release {tag}")
    url = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/releases/assets/{asset['id']}"
    r = requests.get(url, headers={**headers, "Accept": "application/octet-stream"})
    with open(filename, "wb") as f:
        f.write(r.content)
    print(f"✅ Downloaded: {filename}")

download_from_release("word_index.json", current_tag)
download_from_release("last_model.h5", current_tag)

print("📊 Kaggle size:", len(kaggle_df))
print("📥 Firebase size:", len(firebase_df))
print("🔗 Merged dataset size:", len(merged_df))


# ------------------ Tokenizer + Sequence ------------------
with open("word_index.json", "r") as f:
    word_index = json.load(f)

tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
tokenizer.word_index = word_index.copy()
tokenizer.fit_on_texts(texts)
updated_word_index = tokenizer.word_index

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=150)

# ------------------ Train Model ------------------
try:
    model = load_model("last_model.h5")
    print("✅ Loaded previous model.")
except:
    print("🆕 Creating new model...")
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

# ------------------ Upload to GitHub ------------------
payload = {
    "tag_name": TAG,
    "name": f"Model Update {TAG}",
    "body": f"Auto-generated release with {len(merged_df)} samples",
    "draft": False,
    "prerelease": False
}
res = requests.post(f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/releases", headers=headers, json=payload).json()
upload_url = res.get("upload_url", "").split("{")[0]
if not upload_url:
    print("❌ Failed to create release.")
    exit()

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
            print(f"❌ Failed to upload {file_name}: {resp.text}")

for file in ["updated_model.tflite", "last_model.h5", "word_index.json", "model_version.txt"]:
    upload_to_release(file)

print("✅ Finished retraining and uploading.")
