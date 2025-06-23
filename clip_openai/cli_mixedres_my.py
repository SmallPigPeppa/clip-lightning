import time
import random
from datetime import datetime
from tqdm import tqdm  # Install via: pip install tqdm

def simulate_training(total_steps=10, interval_hours=2):
    """
    Simulates a training process with steps.
    Every 'step' is separated by a fixed interval (e.g., 2 hours).
    Randomly generates and prints training loss for each step with a timestamp.
    """
    for step in tqdm(range(1, total_steps + 1), desc="Training Progress", unit="step"):
        # Simulate a random training loss between 0.1 and 1.0
        train_loss = round(random.uniform(0.1, 1.0), 4)

        # Get current timestamp
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Print training info for the current step
        print(f"[{now}] Step {step} | Training Loss: {train_loss}")

        # Wait before the next step (2 hours by default)
        if step != total_steps:
            time.sleep(2 * 60 * 60)  # 2 hours = 7200 seconds
            # time.sleep(5)  # Debug mode: use 5 seconds for testing

# Run the training simulation
simulate_training(total_steps=5)
