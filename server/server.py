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
    
    alpha: weight for system heterogeneity; (1-alpha) weight for statistical heterogeneity.
    """
    if len(client_database) < num_clients_to_select:
        return {}

    available_clients = list(client_database.items())

    # --- Step 1: Compute statistical scores from training loss ---
    stat_scores = [1/(data['training_loss'] + 0.01) for _, data in available_clients]  # inverse loss
    min_s, max_s = min(stat_scores), max(stat_scores)
    stat_scores_norm = [(s - min_s)/(max_s - min_s + 1e-8) for s in stat_scores]  # normalize to 0-1

    # --- Step 2: Compute combined score ---
    combined_scores = {}
    for i, (client_id, data) in enumerate(available_clients):
        combined_score = alpha * data['qos_score'] + (1 - alpha) * stat_scores_norm[i]
        combined_scores[client_id] = combined_score

    # --- Step 3: Select top N clients based on combined score ---
    sorted_clients = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    selected_client_ids = [client_id for client_id, _ in sorted_clients[:num_clients_to_select]]
    print(f"Selected clients based on combined system+statistical utility: {selected_client_ids}")

    # --- Step 4: Dynamic Task Allocation (based on QoS only for now) ---
    tasks = {}
    for client_id in selected_client_ids:
        qos_score = client_database[client_id]['qos_score']

        # Dynamic allocation: higher QoS → more epochs
        if qos_score > 0.8:
            epochs = 10
        elif qos_score > 0.5:
            epochs = 5
        else:
            epochs = 2

        tasks[client_id] = {
            "round_id": 1,  # TODO: implement proper round counter
            "model_weights": weights_to_json(global_model.get_weights()),
            "epochs": epochs
        }
        print(f"  > Assigned task to {client_id}: {epochs} epochs")

    return tasks


# --- 6. Define the API Endpoints ---
@app.post("/register")
def register_client(status: ClientStatus):
    # Now, we also initialize the training_loss for new clients
    client_database[status.client_id] = { 
        "qos_score": status.qos_score,
        "training_loss": 10.0 # Start with a high default loss
    }
    return {"message": "Client registered."}

@app.post("/start-training-round")
def start_training_round(num_clients: int):
    tasks = select_and_assign_tasks(num_clients)
    if not tasks:
        return {"error": "Not enough clients available."}
    return {"message": "Round started.", "tasks": tasks}

@app.post("/submit-update")
def submit_update(update: ModelUpdate):
    print(f"Received update from {update.client_id} for round {update.round_id}")
    
    # Update the client's statistical utility (its loss) in our database
    if update.client_id in client_database:
        client_database[update.client_id]['training_loss'] = update.training_loss

    delta_weights = json_to_weights(update.weight_delta)
    current_weights = global_model.get_weights()
    
    # This is a simplified aggregation for one client.
    # In a real multi-client round, you would average all received deltas.
    new_weights = []
    for i in range(len(current_weights)):
        new_weights.append(current_weights[i] + delta_weights[i])
    global_model.set_weights(new_weights)
    
    print(f"Global model updated by {update.client_id}. New loss for client: {update.training_loss:.4f}")
    return {"status": "update aggregated"}


# --- 7. Main Execution ---
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
