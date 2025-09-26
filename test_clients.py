import threading
import time
import random
import requests
import numpy as np
import tensorflow as tf
from tensorflow import keras

SERVER_URL = "http://localhost:8000"
MAX_ROUNDS = 5

# --- Helper functions ---
def weights_to_json(weights: list) -> list:
    """Converts a list of NumPy arrays to a list of lists (JSON serializable)."""
    return [w.tolist() for w in weights]

def json_to_weights(json_weights: list) -> list:
    """Converts a list of lists back to a list of NumPy arrays."""
    return [np.array(w) for w in json_weights]

# --- Non-IID data mapping and Sample Counts (Increased Client Count) ---
# Simulating statistical heterogeneity (StatHet) by assigning unique/overlapping classes.
client_class_map = {
    "client-001": [0, 1, 2],       # Low StatHet / High Data Utility (Diverse small set)
    "client-002": [3, 4, 5, 6],    # Medium StatHet / Medium Data Utility
    "client-003": [7, 8, 9],       # Low StatHet / High Data Utility (Diverse small set)
    "client-004": [0, 1, 5, 6],    # High StatHet (Mixed classes)
    "client-005": [4, 4, 4, 4],    # Very High StatHet (Only one class - low diversity)
}

client_sample_count = {
    "client-001": 400,
    "client-002": 600,
    "client-003": 300,
    "client-004": 800,
    "client-005": 100,
}

# --- Model Definition (Moved Outside run_client for efficiency) ---
def create_local_model():
    """Defines and compiles the CIFAR-10 CNN model once."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# --- Client logic ---
def run_client(client_id: str):
    print(f"\n--- STARTING CLIENT {client_id} ---")

    # 1. Initialize System and Data
    qos_score = round(random.uniform(0.1, 1.0), 2) # Simulate System Heterogeneity (SysHet)
    print(f"{client_id}: Initial QoS score {qos_score}")

    # Load and prepare local data
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    allowed_classes = client_class_map[client_id]
    
    # Filter data based on assigned classes
    idx = np.isin(y_train, allowed_classes).flatten()
    x_local = x_train[idx].astype("float32") / 255.0
    y_local = y_train[idx]

    sample_count = client_sample_count[client_id]
    
    # Ensure client only uses its allocated sample count
    x_local = x_local[:sample_count] 
    y_local = y_local[:sample_count]
    
    if len(x_local) == 0:
        print(f"{client_id}: No data samples available. Exiting.")
        return

    y_local = keras.utils.to_categorical(y_local, 10)
    local_data_size = len(x_local)

    # Estimate data utility using normalized label entropy
    label_distribution = np.mean(y_local, axis=0)
    # The max possible entropy for 10 classes is ln(10)
    label_entropy = -np.sum(label_distribution * np.log(label_distribution + 1e-8))
    data_utility = round(label_entropy / np.log(10), 2) # Normalize to [0,1]
    print(f"{client_id}: Data Utility (Entropy) {data_utility}. Local Samples: {local_data_size}")

    # 2. Register Client
    try:
        requests.post(f"{SERVER_URL}/register", json={
            "client_id": client_id,
            "qos_score": qos_score,
            "data_utility": data_utility
        }).raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"{client_id}: Registration failed - {e}")
        return

    # 3. Initialize Model ONCE (Efficiency Improvement)
    try:
        local_model = create_local_model()
    except Exception as e:
        print(f"{client_id}: Model initialization failed - {e}")
        return

    # 4. Start FL rounds
    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{client_id}: Starting Round {round_num}")

        # Request Task
        try:
            # We request tasks for 5 clients, hoping to select 5
            response = requests.post(f"{SERVER_URL}/start-training-round?num_clients=5") 
            response.raise_for_status()
            tasks = response.json().get("tasks", {})
            
            if client_id not in tasks:
                print(f"{client_id}: Not selected for Round {round_num}. Waiting...")
                time.sleep(random.uniform(2, 5))
                continue
            
            task = tasks[client_id]
            round_id = task.get("round_id")
            epochs = task.get("epochs")
        except requests.exceptions.RequestException as e:
            print(f"{client_id}: Could not get task - {e}")
            break

        # Training
        original_weights = json_to_weights(task.get("model_weights"))
        local_model.set_weights(original_weights) # Load new global weights

        initial_loss, _ = local_model.evaluate(x_local, y_local, verbose=0)
        print(f"{client_id}: Initial loss: {initial_loss:.4f}. Assigned epochs: {epochs}")

        # Simulate training delay based on QoS (SysHet)
        delay_factor = (1.0 - qos_score) * 5 
        time.sleep(delay_factor + random.uniform(0.1, 0.5))

        # Perform local training
        history = local_model.fit(x_local, y_local, epochs=epochs, batch_size=32, verbose=0)
        final_loss = history.history['loss'][-1]
        print(f"{client_id}: Training complete. Final loss: {final_loss:.4f}")

        # Submit Update
        updated_weights = local_model.get_weights()
        delta_weights = [upd - orig for upd, orig in zip(updated_weights, original_weights)]
        serialized_delta = weights_to_json(delta_weights)

        try:
            payload = {
                "client_id": client_id,
                "round_id": round_id,
                "weight_delta": serialized_delta,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "sample_count": local_data_size
            }
            response = requests.post(f"{SERVER_URL}/submit-update", json=payload)
            response.raise_for_status()
            print(f"{client_id}: Update submitted. Server status: {response.json()['status']}")
        except requests.exceptions.RequestException as e:
            print(f"{client_id}: Could not submit update - {e}")

        time.sleep(random.uniform(0.5, 1.0))

    print(f"--- CLIENT {client_id} FINISHED SIMULATION ---\n")

# --- Run multiple clients concurrently ---
if __name__ == "__main__":
    client_ids = list(client_class_map.keys()) # Automatically includes all 5 clients
    threads = []

    print("Preloading CIFAR-10 data...")
    # This ensures Keras/TF data is ready before threads start
    keras.datasets.cifar10.load_data() 
    print("Data preload complete.")

    for cid in client_ids:
        t = threading.Thread(target=run_client, args=(cid,))
        t.start()
        threads.append(t)
        time.sleep(0.5) # Stagger client starts slightly

    for t in threads:
        t.join()

    print("All clients finished.")
