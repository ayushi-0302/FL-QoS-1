import requests

def fetch_live_status():
    try:
        response = requests.get("http://127.0.0.1:8000/get-status")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error fetching status: {e}")
        return None

def animate_pareto_plot(i, ax):
    """Update function for Matplotlib's animation, plotting QoS vs. Data Utility with clear selection coloring."""
    
    status = fetch_live_status()
    if status is None:
        ax.set_title("FL-QoS Demo: Waiting for Server...", color='gray')
        return

    client_data = status['clients']
    current_round = status['global_round_id']
    selected_clients = status['current_round_clients']
    
    ax.clear()

    if not client_data:
        ax.set_title("FL-QoS Demo: Awaiting Client Registration...", color='blue')
        return

    # Prepare Data
    client_ids = list(client_data.keys())
    qos_scores = [data.get('qos_score', 0.0) for data in client_data.values()]
    data_utilities = [data.get('data_utility', 0.0) for data in client_data.values()]

    # Colors for selection status
    colors = ['gold' if cid in selected_clients else 'dodgerblue' for cid in client_ids]
    alphas = [1.0 if cid in selected_clients else 0.6 for cid in client_ids]

    # Scatter plot
    ax.scatter(data_utilities, qos_scores, 
               c=colors, alpha=alphas, edgecolor='black', linewidth=1.2)

    # Title & labels
    ax.set_title(f"FL-QoS: Client Selection (Round {current_round})", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Data Utility (Entropy/Diversity)", fontsize=12)
    ax.set_ylabel("QoS Score (System Reliability)", fontsize=12)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Annotate with client IDs only
    for x, y, cid in zip(data_utilities, qos_scores, client_ids):
        ax.annotate(cid, (x, y), textcoords="offset points", xytext=(5, 5), 
                    ha='left', fontsize=9, fontweight='bold')

    # Legend
    handle_selected = plt.Line2D([0], [0], marker='o', color='w', label='Selected Client', 
                                 markersize=10, markerfacecolor='gold', markeredgecolor='black', linestyle='')
    handle_unselected = plt.Line2D([0], [0], marker='o', color='w', label='Unselected Client', 
                                   markersize=10, markerfacecolor='dodgerblue', markeredgecolor='black', linestyle='')
    
    ax.legend(handles=[handle_selected, handle_unselected], 
              title="Selection Status",
              loc='upper left', bbox_to_anchor=(1.05, 1.05), fancybox=True, shadow=True)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(right=0.75)  # leave space for legend

    ani = animation.FuncAnimation(fig, animate_pareto_plot, fargs=(ax,), interval=3000)

    print("\n--- Starting FL-QoS Pareto Frontier Visualization ---")
    print("X-axis: Data Utility (StatHet). Y-axis: QoS (SysHet).")
    print("Keep the server and clients running. Plot will update every 3 seconds.")

    plt.show()
