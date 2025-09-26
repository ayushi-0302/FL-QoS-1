# --- Global Configuration ---
MIN_SUBMISSION_RATIO = 0.6
MAX_QOS_EPOCHS = 10
MIN_QOS_EPOCHS = 2

# --- FastAPI Setup ---
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
from typing import List, Dict

# Import TensorFlow/Keras for model definition on the server
import tensorflow as tf
from tensorflow import keras 

app = FastAPI()

# --- In-Memory State ---
global_round_id = 0
client_database = {}
current_round_clients = set()
round_updates = {}

# --- Data Models ---
class ClientStatus(BaseModel):
    client_id: str
    qos_score: float
    data_utility: float  # e.g., entropy, diversity, etc.

class ModelUpdate(BaseModel):
    client_id: str
    round_id: int
    weight_delta: List[list]
    initial_loss: float
    final_loss: float
    sample_count: int

# --- Helper Functions ---
def weights_to_json(weights: List[np.ndarray]) -> List[List[float]]:
    return [w.tolist() for w in weights]

def json_to_weights(json_weights: List[List[float]]) -> List[np.ndarray]:
    return [np.array(w) for w in json_weights]

def calculate_statistical_utility(initial_loss: float, final_loss: float) -> float:
    return max(0.0, initial_loss - final_loss)



# --- Global Model (UPDATED) ---
def create_cifar10_cnn_model():
    """Creates the actual Keras model used by the clients."""
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

class KerasModelWrapper:
    """A simple wrapper to match the get_weights/set_weights interface."""
    def __init__(self, model):
        self.model = model
    def get_weights(self): 
        return self.model.get_weights()
    def set_weights(self, weights): 
        self.model.set_weights(weights)

global_model = KerasModelWrapper(create_cifar10_cnn_model())

# --- Pareto Frontier Selection ---
def pareto_select(clients: List[tuple]) -> List[str]:
    selected = []
    for cid1, data1 in clients:
        dominated = False
        for cid2, data2 in clients:
            if cid1 == cid2: continue
            if (data2['qos_score'] >= data1['qos_score'] and
                data2['data_utility'] >= data1['data_utility'] and
                (data2['qos_score'] > data1['qos_score'] or data2['data_utility'] > data1['data_utility'])):
                dominated = True
                break
        if not dominated:
            selected.append(cid1)
    return selected

# --- Tiered Sampling ---
def assign_tiers(clients: List[str]) -> Dict[str, str]:
    tiers = {}
    for cid in clients:
        qos = client_database[cid]['qos_score']
        data = client_database[cid]['data_utility']
        if qos > 0.7 and data > 0.7:
            tiers[cid] = "Tier 1"
        elif data > 0.7:
            tiers[cid] = "Tier 2"
        elif qos > 0.7:
            tiers[cid] = "Tier 3"
        else:
            tiers[cid] = "Tier 4"
    return tiers

# --- Task Assignment ---
def select_and_assign_tasks(num_clients_to_select: int) -> Dict:
    # Ensure this is the first line to be absolutely safe
    global global_round_id 
    current_rid = global_round_id

    available_clients = list(client_database.items())
    if len(available_clients) < num_clients_to_select:
        return {}

    # Normalize scores
    qos_vals = [data['qos_score'] for _, data in available_clients]
    data_vals = [data['data_utility'] for _, data in available_clients]
    min_q, max_q = min(qos_vals), max(qos_vals)
    min_d, max_d = min(data_vals), max(data_vals)

    for cid, data in client_database.items():
        data['qos_norm'] = (data['qos_score'] - min_q) / (max_q - min_q + 1e-8)
        data['data_norm'] = (data['data_utility'] - min_d) / (max_d - min_d + 1e-8)

    # Pareto selection
    pareto_clients = pareto_select(available_clients)
    tiers = assign_tiers(pareto_clients)

    # Tiered sampling
    selected = []
    for cid in pareto_clients:
        tier = tiers[cid]
        if tier == "Tier 1":
            selected.append(cid)
        elif tier == "Tier 2":
            if np.random.rand() < 0.5:  # lower frequency
                selected.append(cid)
        elif tier == "Tier 3":
            if len(selected) < num_clients_to_select:
                selected.append(cid)

    selected = selected[:num_clients_to_select]
    current_round_clients.update(selected)

    # Assign tasks
    tasks = {}
    for cid in selected:
        qos_norm = client_database[cid]['qos_norm']
        epochs = MIN_QOS_EPOCHS + qos_norm * (MAX_QOS_EPOCHS - MIN_QOS_EPOCHS)
        epochs = max(MIN_QOS_EPOCHS, int(round(epochs)))
        tasks[cid] = {
            "round_id": current_rid,
            "model_weights": weights_to_json(global_model.get_weights()),
            "epochs": epochs
        }

    return tasks

# --- Aggregation ---
def fl_qos_aggregate():
    # Ensure this is the first line to be absolutely safe
    global global_round_id
    current_rid = global_round_id
    updates_list = round_updates.get(current_rid, [])
    if not updates_list: return False

    total_score = sum(u['combined_score'] for u in updates_list)
    
    # Initialize new_weights array based on the current model's structure
    new_weights = [np.zeros_like(w) for w in global_model.get_weights()]

    for u in updates_list:
        weight_factor = u['combined_score'] / (total_score + 1e-8)
        for i, delta in enumerate(u['delta']):
            new_weights[i] += delta * weight_factor

    updated_weights = [cw + nw for cw, nw in zip(global_model.get_weights(), new_weights)]
    global_model.set_weights(updated_weights)

    round_updates[current_rid] = []
    current_round_clients.clear()
    
    # Global keyword used before modification
    global_round_id += 1 
    return True

# --- API Endpoints ---
@app.post("/register")
def register_client(status: ClientStatus):
    client_database[status.client_id] = {
        "qos_score": status.qos_score,
        "data_utility": status.data_utility,
        "training_loss": 10.0
    }
    return {"message": "Client registered."}

@app.post("/start-training-round")
def start_training_round(num_clients: int):
    # This function now only READS the results of select_and_assign_tasks.
    # It does NOT modify global_round_id, so the 'global' keyword is not required 
    # and its omission avoids the tricky parser error.
    
    tasks = select_and_assign_tasks(num_clients)
    if not tasks:
        # If selection fails, the round ID is NOT decremented (no rollback).
        # It remains at the current value, waiting for the next attempt.
        return {"error": "Not enough clients available or zero clients selected in current round."}
    return {"message": "Round started.", "tasks": tasks}

@app.post("/submit-update")
def submit_update(update: ModelUpdate):
    if update.round_id != global_round_id:
        return {"status": "Update discarded (round expired)."}

    stat_utility = calculate_statistical_utility(update.initial_loss, update.final_loss)
    stat_utility_norm = min(1.0, stat_utility)
    qos_score = client_database[update.client_id]['qos_score']
    combined_score = 0.5 * qos_score + 0.5 * stat_utility_norm

    client_database[update.client_id]['training_loss'] = update.final_loss
    current_rid = update.round_id
    if current_rid not in round_updates:
        round_updates[current_rid] = []

    round_updates[current_rid].append({
        "client_id": update.client_id,
        "delta": json_to_weights(update.weight_delta),
        "combined_score": combined_score,
        "n_samples": update.sample_count
    })

    submitted_count = len(round_updates[current_rid])
    if len(current_round_clients) > 0 and submitted_count >= len(current_round_clients) * MIN_SUBMISSION_RATIO:
        fl_qos_aggregate()

    return {"status": "Update recorded."}

@app.get("/get-status")
def get_status():
    return {
        "global_round_id": global_round_id,
        "clients": client_database,
        "current_round_clients": list(current_round_clients),
        "updates_in_round": len(round_updates.get(global_round_id, []))
    }

# --- Main Execution ---
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
