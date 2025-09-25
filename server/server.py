# server.py

# --- Import necessary libraries ---
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random # We need this for random selection during exploration

# --- 1. Initialize the FastAPI application ---
app = FastAPI(
    title="FL-QoS Server",
    description="Manages clients and orchestrates federated learning rounds."
)

# --- 2. In-Memory Storage ---
client_database: Dict[str, Dict] = {}
global_model = None

# --- 3. Define the structure of the data our API expects ---
class ClientStatus(BaseModel):
    client_id: str
    qos_score: float

class ModelUpdate(BaseModel):
    client_id: str
    round_id: int
    weight_delta: List[list]
    # NEW: The client reports its training loss from the last round
    training_loss: float

# --- 4. Create the real global AI model ---
def create_cifar10_cnn_model():
    """Creates a CNN model for CIFAR-10."""
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model

def weights_to_json(weights: List[np.ndarray]) -> List[List[float]]:
    """Serializes NumPy arrays to JSON lists."""
    return [w.tolist() for w in weights]

def json_to_weights(json_weights: List[List[float]]) -> List[np.ndarray]:
    """Deserializes JSON lists back to NumPy arrays."""
    return [np.array(w) for w in json_weights]

global_model = create_cifar10_cnn_model()
print("Initialized the global CIFAR-10 CNN model.")

# --- 5. UPGRADED QOS SELECTION AND TASK ALLOCATION LOGIC ---
def select_and_assign_tasks(num_clients_to_select: int, alpha: float = 0.6) -> Dict:
    """
    FL-QoS client selection considering both system heterogeneity (QoS score)
    and statistical heterogeneity (training loss as proxy).

    Task allocation (epochs) also depends on combined score.
    """
    if len(client_database) < num_clients_to_select:
        return {}

    available_clients = list(client_database.items())

    # --- Step 1: Compute normalized statistical scores ---
    stat_scores = [1 / (data['training_loss'] + 0.01) for _, data in available_clients]
    min_s, max_s = min(stat_scores), max(stat_scores)
    stat_scores_norm = [(s - min_s) / (max_s - min_s + 1e-8) for s in stat_scores]  # normalize 0-1

    # --- Step 2: Compute combined score ---
    combined_scores = {}
    for i, (client_id, data) in enumerate(available_clients):
        combined_score = alpha * data['qos_score'] + (1 - alpha) * stat_scores_norm[i]
        combined_scores[client_id] = combined_score

    # --- Step 3: Select top N clients ---
    sorted_clients = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    selected_client_ids = [client_id for client_id, _ in sorted_clients[:num_clients_to_select]]
    print(f"Selected clients based on combined system+statistical utility: {selected_client_ids}")

    # --- Step 4: Dynamic Task Allocation (based on combined score) ---
    tasks = {}
    max_epochs = 10
    min_epochs = 2

    # Find min/max combined score among selected clients for scaling epochs
    selected_scores = [combined_scores[cid] for cid in selected_client_ids]
    min_score, max_score = min(selected_scores), max(selected_scores)

    for client_id in selected_client_ids:
        # Scale epochs linearly between min_epochs and max_epochs based on combined score
        score = combined_scores[client_id]
        if max_score - min_score < 1e-8:  # avoid division by zero
            epochs = max_epochs
        else:
            epochs = min_epochs + (score - min_score) / (max_score - min_score) * (max_epochs - min_epochs)
            epochs = int(round(epochs))

        tasks[client_id] = {
            "round_id": 1,  # TODO: implement proper round counter
            "model_weights": weights_to_json(global_model.get_weights()),
            "epochs": epochs
        }
        print(f"  > Assigned task to {client_id}: {epochs} epochs (combined score: {combined_scores[client_id]:.3f})")

    return tasks


# --- 6. API Endpoints ---
@app.post("/register")
def register_client(status: ClientStatus):
    client_database[status.client_id] = { 
        "qos_score": status.qos_score,
        "training_loss": 10.0  # Start with a high default loss
    }
    return {"message": "Client registered."}

@app.post("/start-training-round")
def start_training_round(num_clients: int):
    tasks = select_and_assign_tasks(num_clients)
    if not tasks:
        return {"error": "Not enough clients available."}
    return {"message": "Round started.", "tasks": tasks}

# In-memory store for round updates
round_updates: Dict[int, List[Dict]] = {}  # round_id → list of {client_id, delta, combined_score}

@app.post("/submit-update")
def submit_update(update: ModelUpdate):
    print(f"Received update from {update.client_id} for round {update.round_id}")

    # --- Step 1: Update client training loss ---
    if update.client_id in client_database:
        client_database[update.client_id]['training_loss'] = update.training_loss

    # --- Step 2: Compute combined score for weighting ---
    alpha = 0.6
    stat_score = 1 / (update.training_loss + 0.01)
    stat_score_norm = (stat_score - 0) / (1 - 0 + 1e-8)  # normalized 0-1 (simplified)
    combined_score = alpha * client_database[update.client_id]['qos_score'] + (1 - alpha) * stat_score_norm

    # --- Step 3: Store update ---
    if update.round_id not in round_updates:
        round_updates[update.round_id] = []
    round_updates[update.round_id].append({
        "client_id": update.client_id,
        "delta": json_to_weights(update.weight_delta),
        "combined_score": combined_score
    })

    # --- Step 4: Weighted aggregation if all clients submitted ---
    updates_list = round_updates[update.round_id]
    selected_clients_count = len(updates_list)
    # Note: simple check; can be improved with actual selected clients per round
    if selected_clients_count == len([cid for cid in client_database if cid in [u['client_id'] for u in updates_list]]):
        total_score = sum(u['combined_score'] for u in updates_list)
        new_weights = [np.zeros_like(w) for w in global_model.get_weights()]

        for u in updates_list:
            weight_factor = u['combined_score'] / total_score
            for i, delta in enumerate(u['delta']):
                new_weights[i] += delta * weight_factor

        # Update global model
        current_weights = global_model.get_weights()
        updated_weights = [cw + nw for cw, nw in zip(current_weights, new_weights)]
        global_model.set_weights(updated_weights)
        print(f"Global model updated with weighted aggregation for round {update.round_id}")

        # Clear round updates
        round_updates[update.round_id] = []

    return {"status": "update recorded"}

@app.get("/get-status")
def get_status():
    """Returns current round updates and selected clients"""
    return {
        "clients": client_database,
        "round_updates": round_updates
    }

# --- 7. Main Execution ---
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
