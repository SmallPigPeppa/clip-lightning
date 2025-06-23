#!/usr/bin/env python3
import argparse, time, random
from datetime import datetime
from tqdm import tqdm


def simulate_training(steps=5, interval=2):
    """Run dummy training for given steps and hour intervals"""
    for i in tqdm(range(1, steps + 1), "Training", unit="step"):
        loss = round(random.uniform(0.1, 1.0), 4)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Step {i}/{steps} | Loss: {loss}")
        if i < steps:
            time.sleep(interval * 3600)


def main():
    parser = argparse.ArgumentParser(description="Dummy training script")
    parser.add_argument("--total_steps", type=int, default=200,
                        help="Number of steps to simulate")
    parser.add_argument("--interval_hours", type=float, default=2,
                        help="Hours between steps")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring args:", unknown)
    simulate_training(steps=args.total_steps, interval=args.interval_hours)


if __name__ == "__main__":
    main()
