import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import time
from collections import deque
from gym import Env
from gym.spaces import Discrete, Box
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import sklearn
import os
import matplotlib.pyplot as plt

# --- CONSTANTS ---
# Configuration parameters fixed for the real-time simulation
INITIAL_BALANCE = 10000
WINDOW_SIZE = 20

# --- Class Definitions ---

class TradingEnv(Env):
    """Reinforcement Learning Environment for stock trading."""
    def __init__(self, data: pd.DataFrame, scaler_y: sklearn.preprocessing._data.MinMaxScaler, initial_balance=INITIAL_BALANCE):
        super(TradingEnv, self).__init__()
        
        # Data and trading parameters
        self.data = data.reset_index(drop=True)
        self.scaler_y = scaler_y
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        self.done = False
        self.current_action = None

        # Action space: 0 = BUY, 1 = HOLD, 2 = SELL
        self.action_space = Discrete(3)
        self.state_size = len(self.data.columns) + 3
        
        # Observation space: [balance, shares held, net worth, price data...]
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_size,),
            dtype=np.float32,
        )

    def _get_real_close(self, scaled_close):
        """Inverse transform a normalized close price back to its real value."""
        if not isinstance(scaled_close, np.ndarray):
             scaled_close = np.array(scaled_close).reshape(-1, 1)
        # scaler_y was trained on the 'close' price only
        return self.scaler_y.inverse_transform(scaled_close)[0, 0]

    def _get_observation(self):
        """Create the observation vector (still normalized)."""
        # Dataframe columns order: ['open', 'high', 'low', 'volume', 'close']
        obs = np.array([
            self.balance,
            self.shares_held,
            self.net_worth,
            *self.data.iloc[self.current_step].values
        ], dtype=np.float32)
        return obs

    def _take_action(self, action):
        """Execute trading logic (buy, sell, hold)."""
        scaled_close = self.data.iloc[self.current_step]['close']
        current_price = self._get_real_close(scaled_close)
        
        shares_to_buy = 0
        shares_to_sell = 0

        if action == 0:  # BUY
            shares_to_buy = int(self.balance // current_price)
            if shares_to_buy > 0:
                self.shares_held += shares_to_buy
                self.balance -= shares_to_buy * current_price
                st.session_state.log.append(f"BUY {shares_to_buy} shares @ ${current_price:,.2f}")

        elif action == 2:  # SELL
            if self.shares_held > 0:
                shares_to_sell = self.shares_held
                self.balance += self.shares_held * current_price
                self.shares_held = 0
                st.session_state.log.append(f"SELL {shares_to_sell} shares @ ${current_price:,.2f}")

        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(getattr(self, 'max_net_worth', 0), self.net_worth)

    def step(self, action):
        """Execute one step in the environment."""
        prev_net_worth = self.net_worth

        # Take the chosen action
        self._take_action(action)

        # Move to next time step
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        # Reward calculation (not strictly needed for inference mode but kept for completeness)
        reward = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth else 0
        
        obs = self._get_observation()
        return obs, reward, self.done, {}

    def reset(self):
        """Reset environment to the initial state."""
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.current_step = 0
        self.done = False
        self.current_action = None
        return self._get_observation()

class DQNAgentLSTM:
    """DQN Agent using an LSTM model for price prediction."""
    def __init__(self, state_size, action_size, lstm_model_path="lstm_model.h5", dqn_model_path="dqn_model_9.h5", window_size=WINDOW_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.window_size = window_size
        
        # Epsilon is set to 0.0 for pure exploitation
        self.epsilon = 0.0 

        # Load LSTM model 
        try:
            self.lstm_model = load_model(lstm_model_path, compile=False) # compile=False speeds up loading
            self.lstm_model.compile(optimizer='adam', loss='mean_squared_error') # Re-compile
            st.sidebar.success(f"LSTM Model loaded: {lstm_model_path}")
        except Exception as e:
            st.sidebar.error(f"Error loading LSTM model: {e}. Cannot proceed.")
            raise

        # Load DQN model
        try:
            self.model = load_model(dqn_model_path, compile=False)
            self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.001)) # Re-compile
            st.sidebar.success(f"DQN Model loaded: {dqn_model_path}")
        except Exception as e:
            st.sidebar.error(f"Error loading DQN model: {e}. Cannot proceed.")
            raise

    def act(self, augmented_state):
        """
        Acts based on the augmented state (pure exploitation mode).
        """
        if augmented_state.ndim == 1:
            augmented_state = np.expand_dims(augmented_state, axis=0)
            
        # Predict Q-values
        q_values = self.model.predict(augmented_state, verbose=0) 
        return np.argmax(q_values[0])

# --- Data Preprocessing Function ---

@st.cache_data
def load_and_preprocess_data(data_path, scaler_X_path, scaler_y_path):
    """
    Load data and scalers, then apply the same normalization used in the notebook.
    """
    try:
        df = pd.read_csv(data_path)
        
        # Load scalers
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        
        features_X = ['open', 'high', 'low', 'volume']
        feature_y = ['close']
        
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        
        # Ensure correct column order for normalization
        df = df[features_X + feature_y]
        print(df.head())

        # Normalization
        df[feature_y] = scaler_y.transform(df[feature_y])
        df[features_X] = scaler_X.transform(df[features_X])
        
        return df, scaler_y
    
    except Exception as e:
        st.error(f"Error during data/scaler loading or preprocessing: {e}. Ensure all files are present and correct.")
        return None, None

# --- Simulation Logic ---

def get_augmented_state(env, agent, state_history):
    """
    Builds the augmented state (current state + predicted next close price).
    """
    window_size = agent.window_size
    state_size = env.state_size
    
    # Get the current state (based on the current step data)
    current_state = env._get_observation()
    
    # Check if we need to append the state (prevents duplication if not stepped)
    if not state_history or not np.array_equal(state_history[-1], current_state):
        state_history.append(current_state)

    # Pad the sequence if too short
    if len(state_history) < window_size:
        initial_padding_state = state_history[0] if state_history else np.zeros(state_size, dtype=np.float32)
        padded = [initial_padding_state] * (window_size - len(state_history)) + list(state_history)
    else:
        padded = list(state_history)[-window_size:] # Only take the last WINDOW_SIZE
        
    state_seq = np.array(padded).reshape(1, window_size, state_size)
    
    # LSTM Prediction Input: Only the price features 
    lstm_input_indices = [3, 4, 5, 6] 
    lstm_input = state_seq[:, :, lstm_input_indices]
    
    predicted_close = agent.lstm_model.predict(lstm_input, verbose=0).flatten()

    # Augmented State: Last State + Predicted Close
    last_state = state_seq[:, -1, :]
    augmented_state = np.concatenate([last_state, predicted_close.reshape(1, -1)], axis=1)
    
    return augmented_state

def run_step():
    """Runs a single simulation step."""
    env = st.session_state.env
    agent = st.session_state.agent
    
    if env.done:
        st.session_state.log.append("Simulation finished.")
        st.session_state.auto_run = False
        return
        
    if env.current_step < len(env.data) - 1:
        # Get augmented state and action
        augmented_state = get_augmented_state(env, agent, st.session_state.state_history)
        action = agent.act(augmented_state)
        
        # Execute action in environment
        env.step(action)
        
        # Update session state for display
        scaled_close = env.data.iloc[env.current_step]['close']
        real_close = env._get_real_close(scaled_close)
        
        st.session_state.net_worth_history.append(env.net_worth)
        st.session_state.close_price_history.append(real_close)
        st.session_state.actions_history.append({0: "BUY", 1: "HOLD", 2: "SELL"}.get(action, "HOLD"))
        st.session_state.current_action = {0: "BUY", 1: "HOLD", 2: "SELL"}.get(action, "HOLD")

        st.session_state.current_step_index = env.current_step
    else:
        env.done = True
        st.session_state.log.append("Simulation finished.")
        st.session_state.auto_run = False
    
    # Rerun if auto-run is active
    if st.session_state.auto_run:
        time.sleep(1) # Simulate 1 second delay
        st.rerun()

# --- Streamlit Interface ---

def main():
    st.set_page_config(
        page_title="DQNAgentLSTM Trading",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("DQNAgentLSTM Trading Simulation")
    st.caption("A Deep Q-Network Agent augmented with an LSTM price predictor to make trading decisions (BUY/HOLD/SELL).")

    # --- Sidebar Configuration and Status ---
    st.sidebar.header("Configuration & Status")

    # File names
    data_file = 'AMZN_1min_firstratedata.csv'
    scaler_X_file = 'scaler_X.pkl'
    scaler_y_file = 'scaler_y.pkl'
    lstm_model_file = 'lstm_model.h5'
    dqn_model_file = 'dqn_model_9.h5' 
    
    st.sidebar.subheader("Fixed Parameters")
    st.sidebar.info(f"Initial Balance: **${INITIAL_BALANCE:,.2f}**")
    st.sidebar.info(f"LSTM Window Size: **{WINDOW_SIZE}** steps")

    st.sidebar.subheader("Required Files Status")
    
    files_ok = True
    for f in [data_file, scaler_X_file, scaler_y_file, lstm_model_file, dqn_model_file]:
        if not os.path.exists(f):
            st.sidebar.error(f"❌ Missing: {f}")
            files_ok = False


    # Initialize auto_run flag if not present
    if 'auto_run' not in st.session_state:
        st.session_state.auto_run = False

    # --- Initialization Logic ---
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        if files_ok:
            try:
                # Load and Preprocess
                df_amazon, scaler_y = load_and_preprocess_data(data_file, scaler_X_file, scaler_y_file)
                
                # Use last 1000 rows for demo
                df_amazon = df_amazon.iloc[-1000:].copy() 
                
                # Initialize Environment and Agent
                env = TradingEnv(data=df_amazon, scaler_y=scaler_y, initial_balance=INITIAL_BALANCE)
                agent = DQNAgentLSTM(
                    state_size=env.observation_space.shape[0], 
                    action_size=env.action_space.n,
                    lstm_model_path=lstm_model_file,
                    dqn_model_path=dqn_model_file,
                    window_size=WINDOW_SIZE
                )
                
                # Store in session state
                st.session_state.env = env
                st.session_state.agent = agent
                st.session_state.initialized = True
                st.session_state.auto_run = False
                
                # Initialize simulation history
                env.reset()
                st.session_state.log = deque(maxlen=20)
                st.session_state.current_step_index = env.current_step
                st.session_state.net_worth_history = [env.initial_balance]
                st.session_state.close_price_history = [env._get_real_close(env.data.iloc[env.current_step]['close'])]
                st.session_state.actions_history = ['HOLD']
                st.session_state.state_history = deque(maxlen=agent.window_size)
                st.session_state.current_action = "HOLD"
                st.session_state.log.append("Application ready. Click 'Next Step' or 'Start Auto-Run'.")
            except Exception as e:
                 st.session_state.initialized = False
        else:
            st.warning("Please ensure all required model and data files are present to start the simulation.")

    # --- Main Application Body ---
    if st.session_state.get('initialized', False):
        env = st.session_state.env

        # Prominent Action Display
        action_color = {
            "BUY": "green", 
            "SELL": "red", 
            "HOLD": "gray",
            "INITIALIZING": "orange"
        }.get(st.session_state.current_action, "gray")
        
        st.markdown(
            f"<h2 style='text-align: center; color: {action_color};'>ACTION: {st.session_state.current_action}</h2>", 
            unsafe_allow_html=True
        )
        st.markdown("---")

        # Control Buttons
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1.5])
        
        # Next Step Button
        with col1:
            if env.done:
                 st.button("Next Step (1min)", disabled=True)
                 st.write("**:red[SIMULATION FINISHED]**")
            elif st.button("Next Step (1min)", disabled=st.session_state.auto_run):
                run_step()
                
        # Auto-Run Button
        with col2:
            if st.session_state.auto_run:
                if st.button("Stop Auto-Run"):
                    st.session_state.auto_run = False
                    st.toast("Auto-Run Stopped.")
                    st.rerun() # Stop the continuous rerun
            elif st.button("Start Auto-Run", disabled=env.done):
                st.session_state.auto_run = True
                st.toast("Auto-Run Started. Steps every 1 second.")
                st.rerun()
                
        # Reset Button
        with col3:
            if st.button("Reset Simulation"):
                st.session_state.initialized = False # Force re-initialization
                st.session_state.auto_run = False
                st.rerun()

        # Step Counter
        with col4:
            st.metric("Total Timesteps", f"{env.current_step} / {len(env.data)}")


        # Key Metrics
        col_metrics = st.columns(4)
        
        current_close = st.session_state.close_price_history[-1]
        net_worth_change = (env.net_worth - env.initial_balance) / env.initial_balance
        
        col_metrics[0].metric("Current Net Worth", f"${env.net_worth:,.2f}", f"{net_worth_change:.2%}")
        col_metrics[1].metric("Cash Balance", f"${env.balance:,.2f}")
        col_metrics[2].metric("Shares Held", f"{env.shares_held}")
        col_metrics[3].metric("Current Close Price", f"${current_close:,.2f}")

        # Trading Chart
        st.subheader("Net Worth and Price Evolution")
        
        net_worth_df = pd.DataFrame({
            'Net Worth': st.session_state.net_worth_history,
            'Close Price': st.session_state.close_price_history
        })
        
        # Prepare action markers using Matplotlib standard markers ('^' for buy, 'v' for sell)
        action_markers = []
        for i, action in enumerate(st.session_state.actions_history):
            if action == 'BUY':
                action_markers.append((i, st.session_state.close_price_history[i], 'green', '^')) 
            elif action == 'SELL':
                action_markers.append((i, st.session_state.close_price_history[i], 'red', 'v')) 
                
        
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Plot Net Worth (Left Axis)
        color = 'tab:blue'
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Net Worth ($)', color=color)
        ax1.plot(net_worth_df.index, net_worth_df['Net Worth'], color=color, label='Net Worth')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Plot Close Price (Right Axis)
        ax2 = ax1.twinx()  
        color = 'tab:orange'
        ax2.set_ylabel('Close Price ($)', color=color)  
        ax2.plot(net_worth_df.index, net_worth_df['Close Price'], color=color, linestyle='--', alpha=0.6, label='Close Price')
        ax2.tick_params(axis='y', labelcolor=color)

        # Add Action Markers
        for step, price, c, marker in action_markers:
            ax2.scatter(step, price, color=c, marker=marker, s=100) 

        # Final check for auto-run state to execute the next step
        if st.session_state.auto_run and not env.done:
            # This triggers the automatic loop
            st.pyplot(fig) # Display the chart before the delay
            run_step()
        else:
            st.pyplot(fig)

        # Log
        st.subheader("Recent Action Log")
        log_display = ""
        for entry in reversed(st.session_state.log):
            if 'BUY' in entry or 'SELL' in entry or 'ready' in entry:
                log_display += f"**{entry}** \n"
            else:
                log_display += f"{entry} \n"
        
        st.markdown(log_display)
        
    
if __name__ == "__main__":
    main()
