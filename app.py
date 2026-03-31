import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="Gambler's Ruin Simulator", layout="wide")

st.title("🎰 Gambler's Ruin: European Roulette Analysis")
st.markdown("Explore the probabilities of ruin using Markov Chain formulas and Monte Carlo simulations.")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Parameters")

p_exact = 18 / 37  # Probability of winning (~48.648%)
q_exact = 19 / 37  # Probability of losing (~51.351%)

base_k = st.sidebar.slider("Starting Bankroll (k)", min_value=10, max_value=200, value=100, step=10)
base_N = st.sidebar.slider("Target Bankroll (N)", min_value=base_k+10, max_value=500, value=200, step=10)
simulations = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)

# --- Core Functions ---
def prob_ruin(k, N, p):
    """Calculates the exact probability of ruin using Markov Chain formulas."""
    if p == 0.5:
        return 1 - (k / N)
    else:
        q = 1 - p
        prob_success = (1 - (q/p)**k) / (1 - (q/p)**N)
        return 1 - prob_success

@st.cache_data
def simulate_gamblers_ruin(k_start, N_target, p_win, num_simulations):
    ruin_count = 0
    ruin_times = []
    
    for _ in range(num_simulations):
        bankroll = k_start
        steps = 0
        
        while bankroll > 0 and bankroll < N_target and steps < 5000:
            if np.random.rand() < p_win:
                bankroll += 1
            else:
                bankroll -= 1
            steps += 1
            
        if bankroll == 0:
            ruin_count += 1
            ruin_times.append(steps)
            
    simulated_prob_ruin = ruin_count / num_simulations
    return simulated_prob_ruin, ruin_times

# --- Calculations ---
theoretical_ruin = prob_ruin(base_k, base_N, p_exact)

with st.spinner('Running simulations...'):
    simulated_ruin, ruin_times = simulate_gamblers_ruin(base_k, base_N, p_exact, simulations)

# --- Display Metrics ---
st.subheader("Key Statistics")
col1, col2, col3, col4 = st.columns(4)

house_edge = ((18/37*1) + (19/37*-1))*100
ratio = q_exact / p_exact
diff = abs(theoretical_ruin - simulated_ruin) * 100

col1.metric("House Edge (Expected Value)", f"{house_edge:.2f}%")
col2.metric("Ratio (q/p)", f"{ratio:.4f}")
col3.metric("Theoretical Ruin", f"{theoretical_ruin*100:.2f}%")
col4.metric(f"Simulated Ruin ({simulations} runs)", f"{simulated_ruin*100:.2f}%", delta=f"{diff:.4f}% diff", delta_color="inverse")

st.divider()

# --- Table Generation ---
st.subheader(f"Probability Data Table (Target N={base_N})")
table_data = []
for test_k in [10, 25, 50, 75, 90]:
    r_prob = prob_ruin(test_k, base_N, p_exact)
    table_data.append({"Starting Capital (k)": test_k, "P(Ruin) %": round(r_prob*100, 2), "P(Win) %": round((1-r_prob)*100, 2)})

df = pd.DataFrame(table_data)
st.dataframe(df, use_container_width=True)

st.divider()

# --- Matplotlib Visualizations ---
st.subheader("The Four Pillars of the Gambler's Ruin")

plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(2, 2, figsize=(16, 10))

# 1. House Edge
p_array = np.linspace(0.40, 0.60, 100)
ruin_p = [prob_ruin(base_k, base_N, p_val) * 100 for p_val in p_array]
axs[0, 0].plot(p_array, ruin_p, color='crimson', linewidth=3)
axs[0, 0].axvline(0.5, color='gray', linestyle='--', label='Fair Game (p=0.5)')
axs[0, 0].axvline(p_exact, color='indigo', linestyle='--', label=f'Roulette (p≈0.486)')
axs[0, 0].set_title("1. Impact of the House Edge (p vs q)", fontsize=14, fontweight='bold', color='crimson')
axs[0, 0].set_xlabel("Probability of Winning a Single Hand (p)")
axs[0, 0].set_ylabel("Probability of Ruin (%)")
axs[0, 0].legend()

# 2. Starting Capital
k_array = np.arange(1, 100)
ruin_k = [prob_ruin(k_val, base_N, p_exact) * 100 for k_val in k_array]
axs[0, 1].plot(k_array, ruin_k, color='navy', linewidth=3)
axs[0, 1].axvline(base_k, color='orange', linestyle='--', label=f'Start = {base_k}')
axs[0, 1].set_title("2. Starting Capital (k) is Nonlinear", fontsize=14, fontweight='bold', color='navy')
axs[0, 1].set_xlabel("Starting Bankroll (k)")
axs[0, 1].set_ylabel("Probability of Ruin (%)")
axs[0, 1].legend()

# 3. Target N
N_array = np.arange(60, 201, 2)
ruin_N = [prob_ruin(base_k, N_val, p_exact) * 100 for N_val in N_array]
axs[1, 0].plot(N_array, ruin_N, color='royalblue', linewidth=3)
axs[1, 0].axvline(base_N, color='orange', linestyle='--', label=f'Target = {base_N}')
axs[1, 0].set_title("3. The Danger of a Distant Target (N)", fontsize=14, fontweight='bold', color='royalblue')
axs[1, 0].set_xlabel("Target Bankroll (N)")
axs[1, 0].set_ylabel("Probability of Ruin (%)")
axs[1, 0].legend()

# 4. Number of Hands
if ruin_times:
    sorted_times = np.sort(ruin_times)
    cumulative_ruin = np.arange(1, len(sorted_times) + 1) / simulations * 100
    axs[1, 1].plot(sorted_times, cumulative_ruin, color='mediumseagreen', linewidth=3)
    axs[1, 1].set_title("4. Time in System (Law of Large Numbers)", fontsize=14, fontweight='bold', color='mediumseagreen')
    axs[1, 1].set_xlabel("Number of Hands Played Before Ruin")
    axs[1, 1].set_ylabel("Cumulative Players Ruined (%)")
    axs[1, 1].set_xlim(0, max(3000, max(sorted_times) if len(sorted_times) > 0 else 3000))
else:
    axs[1, 1].text(0.5, 0.5, "No ruins occurred in simulation", ha='center', va='center')
    axs[1, 1].set_title("4. Time in System (Law of Large Numbers)", fontsize=14, fontweight='bold', color='mediumseagreen')

plt.tight_layout()
st.pyplot(fig)
