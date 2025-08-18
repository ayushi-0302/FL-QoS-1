# test_client.py

import requests
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

# The address of the running server
SERVER_URL = "http://localhost:8000"

# --- Helper functions to handle model weights ---
def weights_to_json(weights: list) -> list:
    """Serializes a list of NumPy arrays into a JSON-compatible list of lists."""
    return [w.tolist() for w in weights]

def json_to_weights(json_weights: list) -> list:
    """Deserializes a JSON list of lists back into a list of NumPy arrays."""
    return [np.array(w) for w in json_weights]

# --- Main Client Simulation Logic ---
def run_client(client_id: str):
    """A full simulation of one client performing a training round."""
    
    print(f"\n--- STARTING CLIENT {client_id} ---")

    # 1. Register with the server to report QoS
    qos_score = round(random.uniform(0.1, 1.0), 2)
    print(f"Step 1: Registering with QoS score: {qos_score}")
    try:
        requests.post(f"{SERVER_URL}/register", json={"client_id": client_id, "qos_score": qos_score})
    except requests.exceptions.RequestException as e:
        print(f"  > Could not register. Error: {e}")
        return

    # 2. Request a training task from the server
    print("Step 2: Requesting a training task...")
    # ** NEW ** We now ask the server to start a round and assign us a task
    try:
        # For a simple test, we ask the server to select 1 client.
        response = requests.post(f"{SERVER_URL}/start-training-round?num_clients=1")
        response.raise_for_status()
        # The server now sends back a dictionary of tasks
        tasks = response.json().get("tasks", {})
        if client_id not in tasks:
            print(f"  > Client {client_id} was not selected for this round. Exiting.")
            return
        task = tasks[client_id]
        print("  > Task received.")
    except requests.exceptions.RequestException as e:
        print(f"  > Could not get task. Error: {e}")
        return

    # 3. Prepare for training
    original_weights = json_to_weights(task.get("model_weights"))
    epochs = task.get("epochs")
    
    local_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    local_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    local_model.set_weights(original_weights)

    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    x_local = x_train[:500].astype("float32") / 255.0
    y_local = keras.utils.to_categorical(y_train[:500], 10)

    # 4. Perform local training
    print(f"Step 3: Starting local training for {epochs} epochs...")
    # ** NEW ** We capture the 'history' object to get the training loss
    history = local_model.fit(x_local, y_local, epochs=epochs, batch_size=32, verbose=0)
    final_loss = history.history['loss'][-1]
    print(f"  > Local training complete. Final loss: {final_loss:.4f}")

    # 5. Calculate the weight delta
    updated_weights = local_model.get_weights()
    delta_weights = [updated - original for original, updated in zip(original_weights, updated_weights)]
    serialized_delta = weights_to_json(delta_weights)

    # 6. Submit the update to the server
    print("Step 4: Submitting model update to server...")
    try:
        update_payload = {
            "client_id": client_id,
            "round_id": task.get("round_id"),
            "weight_delta": serialized_delta,
            "training_loss": final_loss # ** NEW ** We include the final loss
        }
        response = requests.post(f"{SERVER_URL}/submit-update", json=update_payload)
        response.raise_for_status()
        print(f"  > Update submitted successfully. Server says: {response.json()['status']}")
    except requests.exceptions.RequestException as e:
        print(f"  > Could not submit update. Error: {e}")
        
    print(f"--- CLIENT {client_id} FINISHED ---")


if __name__ == "__main__":
    # We will simulate a pool of clients registering first
    client_pool = ["client-001", "client-002", "client-003", "client-004"]
    for client_id in client_pool:
        qos = round(random.uniform(0.1, 1.0), 2)
        print(f"Registering {client_id} with QoS: {qos}")
        requests.post(f"{SERVER_URL}/register", json={"client_id": client_id, "qos_score": qos})
        time.sleep(0.5)

    # Now, run a full cycle for one of the clients.
    # The server will select from the pool of registered clients.
    run_client(client_id="client-001")