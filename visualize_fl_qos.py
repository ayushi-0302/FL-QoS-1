import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import requests
import numpy as np

SERVER_URL = "http://localhost:8000"

fig, ax = plt.subplots(figsize=(10,6))

def fetch_status():
    """Fetch the current status from the server."""
    try:
        response = requests.get(f"{SERVER_URL}/get-status")
        data = response.json()
        return data
    except:
        return {"clients": {}, "round_updates": {}}

def update(frame):
    ax.clear()
    status = fetch_status()
    clients = status.get("clients", {})
    round_updates = status.get("round_updates", {})

    client_ids = list(clients.keys())
    qos_values = []
    stat_values = []
    combined_scores = []
    colors = []

    # Calculate scores for stacked bar visualization
    for cid in client_ids:
        qos = clients[cid]["qos_score"]
        training_loss = clients[cid].get("training_loss", 10.0)
        stat_score = 1 / (training_loss + 0.01)
        stat_score_norm = (stat_score - 0) / (1 - 0 + 1e-8)
        combined = 0.6 * qos + 0.4 * stat_score_norm

        qos_values.append(0.6 * qos)
        stat_values.append(0.4 * stat_score_norm)
        combined_scores.append(combined)

        # Highlight selected clients
        selected = cid in round_updates
        colors.append("limegreen" if selected else "lightgray")

    bars_qos = ax.bar(client_ids, qos_values, color='skyblue', label='QoS contribution')
    bars_stat = ax.bar(client_ids, stat_values, bottom=qos_values, color='orange', label='Statistical contribution')

    # Overlay highlight for selected clients
    for i, cid in enumerate(client_ids):
        if cid in round_updates:
            bars_stat[i].set_edgecolor('red')
            bars_stat[i].set_linewidth(2)

    # Add numeric labels on top of bars
    for i, val in enumerate(combined_scores):
        ax.text(i, qos_values[i]+stat_values[i]+0.02, f"{val:.2f}", ha='center', va='bottom', fontweight='bold')

    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Combined Score")
    ax.set_title("FL-QoS: Client Selection & Contribution (Real-Time)")
    ax.legend()

ani = FuncAnimation(fig, update, interval=1000)
plt.show()
