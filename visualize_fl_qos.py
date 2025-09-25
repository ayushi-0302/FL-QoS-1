import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import requests

SERVER_URL = "http://localhost:8000"

fig, ax = plt.subplots(figsize=(10, 6))

# Store previous combined scores for smooth transitions
previous_scores = {}
training_done = False  # flag to freeze once finished

def fetch_status():
    """Fetch the current status from the server."""
    try:
        response = requests.get(f"{SERVER_URL}/get-status")
        data = response.json()
        return data
    except:
        return {"clients": {}, "round_updates": {}, "finished": False}

def update(frame):
    global previous_scores, training_done

    status = fetch_status()
    clients = status["clients"]
    client_ids = list(clients.keys())

    if status.get("finished", False):
        training_done = True

    if not client_ids:
        ax.clear()
        ax.set_title("Waiting for client updates...")
        return

    qos_scores = [clients[c]["qos_score"] for c in client_ids]
    stat_scores_raw = [1 / (clients[c].get("training_loss", 10.0) + 0.01) for c in client_ids]
    max_stat_score = max(stat_scores_raw) if stat_scores_raw else 1.0
    stat_scores = [s / max_stat_score for s in stat_scores_raw]

    combined_scores = [0.6*q + 0.4*s for q, s in zip(qos_scores, stat_scores)]

    # Freeze scores if training is done
    if training_done:
        smooth_scores = combined_scores
    else:
        smooth_scores = []
        for cid, score in zip(client_ids, combined_scores):
            prev = previous_scores.get(cid, 0)
            smooth = prev + 0.15 * (score - prev)  # gradual transition
            smooth_scores.append(smooth)
            previous_scores[cid] = smooth

    # Sort by combined score (highest first)
    sorted_clients = sorted(zip(client_ids, qos_scores, stat_scores, smooth_scores),
                            key=lambda x: x[3], reverse=True)
    client_ids, qos_scores, stat_scores, smooth_scores = zip(*sorted_clients)

    ax.clear()

    # QoS contribution bars
    bottoms = [0]*len(client_ids)
    ax.bar(client_ids, [0.6*q for q in qos_scores], bottom=bottoms,
           color="#1f77b4", alpha=0.7, label="QoS Contribution")

    # Stat contribution bars
    bottoms = [0.6*q for q in qos_scores]
    ax.bar(client_ids, [0.4*s for s in stat_scores], bottom=bottoms,
           color="#ff7f0e", alpha=0.7, label="Stat Contribution")

    # Overlay combined score line
    ax.plot(client_ids, smooth_scores, marker="o", color="black",
            linewidth=2, label="Combined Score")

    # Score labels
    for i, cid in enumerate(client_ids):
        ax.text(i, smooth_scores[i] + 0.02, f"{smooth_scores[i]:.2f}",
                ha='center', va='bottom', fontsize=9, color="black")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Combined Score")
    title = "FL-QoS: Real-Time Client Contribution"
    if training_done:
        title += " (Finalized)"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

ani = FuncAnimation(fig, update, interval=1000)
plt.tight_layout()
plt.show()
