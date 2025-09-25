import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
import random

# --- Mock data store (replace with server data or API calls) ---
client_database = {
    "client-001": {"qos_score": 0.9, "training_loss": 1.2},
    "client-002": {"qos_score": 0.7, "training_loss": 2.0},
    "client-003": {"qos_score": 0.5, "training_loss": 0.8}
}

round_updates = {}

# --- Function to simulate updates for demo ---
def simulate_round():
    for round_id in range(1, 6):
        print(f"\n--- Round {round_id} ---")
        updates = []
        for cid, data in client_database.items():
            # Randomly simulate training loss improvement
            new_loss = max(0.1, data['training_loss'] * random.uniform(0.7, 0.95))
            data['training_loss'] = new_loss
            # Compute combined score (alpha=0.6)
            stat_score = 1 / (new_loss + 0.01)
            stat_score_norm = (stat_score - 0) / (1 - 0 + 1e-8)
            combined_score = 0.6 * data['qos_score'] + 0.4 * stat_score_norm
            updates.append({"client_id": cid, "combined_score": combined_score, "training_loss": new_loss})
        round_updates[round_id] = updates
        time.sleep(1)  # simulate time between rounds

# --- Visualization ---
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(list(client_database.keys()), [0]*len(client_database), color='skyblue')
ax.set_ylim(0, 1.2)
ax.set_ylabel("Combined Score / Epoch Weight")
ax.set_title("FL-QoS Client Selection and Contribution")

def update(frame):
    round_id = frame + 1
    if round_id not in round_updates:
        return
    updates = round_updates[round_id]
    scores = [u['combined_score'] for u in updates]
    for bar, score in zip(bars, scores):
        bar.set_height(score)
        bar.set_color('green' if score > 0.6 else 'orange')  # highlight high contribution
    ax.set_title(f"Round {round_id}: Combined Scores / Epoch Allocation")

# --- Run simulation in background thread ---
sim_thread = threading.Thread(target=simulate_round)
sim_thread.start()

ani = FuncAnimation(fig, update, frames=5, interval=1500, repeat=False)
plt.show()
