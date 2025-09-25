import threading
import time
import random
import requests
import numpy as np
import tensorflow as tf
from tensorflow import keras

SERVER_URL = "http://localhost:8000"

# --- Helper functions ---
def weights_to_json(weights: list) -> list:
    return [w.tolist() for w in weights]

def json_to_weights(json_weights: list) -> list:
    return [np.array(w) for w in json_weights]

# --- Non-IID data mapping ---
client_class_map = {
    "client-001": [0,1,2],       # Classes 0,1,2
    "client-002": [3,4,5,6],     # Classes 3,4,5,6
    "client-003": [7,8,9]        # Classes 7,8,9
}
client_sample_count = {
    "client-001": 400,
    "client-002": 600,
    "client-003": 300
}

# --- Client logic ---
def run_client(client_id: str):
    print(f"\n--- STARTING CLIENT {client_id} ---")

    # 1. Register client
    qos_score = round(random.uniform(0.1, 1.0), 2)
    print(f"{client_id}: Registering with QoS score {qos_score}")
    try:
        requests.post(f"{SERVER_URL}/register", json={"client_id": client_id, "qos_score": qos_score})
    except requests.exceptions.RequestException as e:
        print(f"{client_id}: Registration failed - {e}")
        return

    # 2. Request training task
    try:
        response = requests.post(f"{SERVER_URL}/start-training-round?num_clients=3")
        response.raise_for_status()
        tasks = response.json().get("tasks", {})
        if client_id not in tasks:
            print(f"{client_id}: Not selected for this round.")
            return
        task = tasks[client_id]
    except requests.exceptions.RequestException as e:
        print(f"{client_id}: Could not get task - {e}")
        return

    # 3. Prepare local model
    original_weights = json_to_weights(task.get("model_weights"))
    epochs = task.get("epochs")

    local_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    local_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    local_model.set_weights(original_weights)

    # Load CIFAR-10 and select non-IID subset
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    allowed_classes = client_class_map[client_id]
    idx = np.isin(y_train, allowed_classes).flatten()
    x_local = x_train[idx].astype("float32") / 255.0
    y_local = y_train[idx]
    
    # Limit samples per client for heterogeneity
    sample_count = client_sample_count[client_id]
    x_local = x_local[:sample_count]
    y_local = y_local[:sample_count]
    y_local = keras.utils.to_categorical(y_local, 10)

    # 4. Train locally
    print(f"{client_id}: Training for {epochs} epochs on {len(x_local)} samples...")
    history = local_model.fit(x_local, y_local, epochs=epochs, batch_size=32, verbose=0)
    final_loss = history.history['loss'][-1]
    print(f"{client_id}: Training complete. Final loss: {final_loss:.4f}")

    # 5. Compute delta and submit
    updated_weights = local_model.get_weights()
    delta_weights = [upd - orig for upd, orig in zip(updated_weights, original_weights)]
    serialized_delta = weights_to_json(delta_weights)

    try:
        payload = {
            "client_id": client_id,
            "round_id": task.get("round_id"),
            "weight_delta": serialized_delta,
            "training_loss": final_loss
        }
        response = requests.post(f"{SERVER_URL}/submit-update", json=payload)
        response.raise_for_status()
        print(f"{client_id}: Update submitted. Server status: {response.json()['status']}")
    except requests.exceptions.RequestException as e:
        print(f"{client_id}: Could not submit update - {e}")

    print(f"--- CLIENT {client_id} FINISHED ---\n")

# --- Run multiple clients concurrently ---
if __name__ == "__main__":
    client_ids = ["client-001", "client-002", "client-003"]
    threads = []

    for cid in client_ids:
        t = threading.Thread(target=run_client, args=(cid,))
        t.start()
        threads.append(t)
        time.sleep(0.2)  # slight stagger to simulate real network delays

    for t in threads:
        t.join()

    print("All clients finished.")
