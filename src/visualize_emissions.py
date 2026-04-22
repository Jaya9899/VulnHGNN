
import json
import matplotlib.pyplot as plt
import os
import sys

def visualize_carbon(report_path="results/carbon_report.json"):
    if not os.path.exists(report_path):
        print(f"[!] Report not found at {report_path}")
        return

    with open(report_path, "r") as f:
        data = json.load(f)

    phases = list(data["phases"].keys())
    durations = [p["duration_sec"] for p in data["phases"].values()]
    energies = [p["energy_kwh"] for p in data["phases"].values()]
    co2 = [p["co2_kg"] for p in data["phases"].values()]

    # Shorten phase names for labels
    labels = [p.replace("inference_", "") for p in phases]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Duration Plot
    ax1.bar(labels, durations, color='skyblue')
    ax1.set_title('Duration per Phase (seconds)')
    ax1.set_ylabel('Seconds')

    # Energy Plot
    ax2.bar(labels, energies, color='lightgreen')
    ax2.set_title('Energy Consumption per Phase (kWh)')
    ax2.set_ylabel('kWh')

    # CO2 Plot
    ax3.bar(labels, co2, color='salmon')
    ax3.set_title('CO2 Emissions per Phase (kg)')
    ax3.set_ylabel('kg CO2')

    plt.tight_layout()
    output_path = "results/carbon_emissions_visual.png"
    plt.savefig(output_path)
    print(f"[+] Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_carbon()
