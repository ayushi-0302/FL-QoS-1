import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import requests
import time

SERVER_URL = "http://localhost:8000"

fig, ax = plt.subplots(figsize=(8,5))

def fetch_status():
    try:
        response = requests.get(f"{SERVER_URL}/get-status")
        data = response.json()
        return data
    except:
        return {"clients": {}, "round_updates": {}}

def update(frame):
    status = fetch_status()
    clients = status["clients"]
    round_updates = status["round_updates"]

    client_ids = list(clients.keys())
    combined_scores = []
    colors = []

    # Calculate combined score for each client
    for cid in client_ids:
        qos = clients[cid]["qos_score"]
        training_loss = clients[cid].get("training_loss", 10.0)
        stat_score = 1 / (training_loss + 0.01)
        stat_score_norm = (stat_score - 0) / (1 - 0 + 1e-8)
        combined = 0.6 * qos + 0.4 * stat_score_norm
        combined_scores.append(combined)
        colors.append("green" if combined > 0.6 else "orange")

    ax.clear()
    bars = ax.bar(client_ids, combined_scores, color=colors)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Combined Score")
    ax.set_title("FL-QoS: Real-Time Client Selection & Contribution")

ani = FuncAnimation(fig, update, interval=1000)
plt.show()
