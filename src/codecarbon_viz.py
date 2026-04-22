
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_codecarbon_dashboard(csv_path="results/emissions.csv"):
    if not os.path.exists(csv_path):
        print(f"[!] Emissions file not found at {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)
    
    # Preprocessing: Shorten task names for better labels
    df['task'] = df['project_name'].apply(lambda x: x.split('_')[-1])
    
    # Set style to something more "dashboard-like"
    plt.style.use('ggplot')
    sns.set_palette("viridis")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('VulnHGNN Sustainability Dashboard (CodeCarbon)', fontsize=22, fontweight='bold', y=0.95)

    # 1. Total Energy Consumption by Task
    sns.barplot(x='task', y='energy_consumed', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Energy Consumed (kWh)', fontsize=16)
    axes[0, 0].set_ylabel('kWh')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. CO2 Emissions by Task
    sns.barplot(x='task', y='emissions', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('CO2 Emissions (kg)', fontsize=16)
    axes[0, 1].set_ylabel('kg CO2')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Power Usage Breakdown (CPU vs GPU if available)
    # Most Windows environments without specific drivers will show CPU only
    power_cols = ['cpu_power', 'gpu_power', 'ram_power']
    available_power = [col for col in power_cols if col in df.columns and df[col].sum() > 0]
    
    if available_power:
        df_power = df.groupby('task')[available_power].mean()
        df_power.plot(kind='bar', stacked=True, ax=axes[1, 0])
        axes[1, 0].set_title('Mean Power Draw (Watts)', fontsize=16)
        axes[1, 0].set_ylabel('Watts')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend(title="Component")
    else:
        axes[1, 0].text(0.5, 0.5, 'Hardware Power Detail Unavailable\n(Requires Intel Power Gadget or NVIDIA-SMI)', 
                       ha='center', va='center', fontsize=12, color='gray')
        axes[1, 0].set_title('Hardware Power Detail', fontsize=16)

    # 4. Duration vs Emissions (Scatter)
    sns.scatterplot(x='duration', y='emissions', size='energy_consumed', data=df, ax=axes[1, 1], sizes=(100, 500))
    axes[1, 1].set_title('Efficiency: Duration vs Emissions', fontsize=16)
    axes[1, 1].set_xlabel('Duration (seconds)')
    axes[1, 1].set_ylabel('kg CO2')

    # Add summary statistics text box
    total_co2 = df['emissions'].sum()
    total_energy = df['energy_consumed'].sum()
    avg_cpu = df['cpu_power'].mean() if 'cpu_power' in df.columns else 0
    
    stats_text = (
        f"TOTAL METRICS\n"
        f"-----------------\n"
        f"Total CO2:    {total_co2:.8f} kg\n"
        f"Total Energy: {total_energy:.8f} kWh\n"
        f"Avg CPU Power: {avg_cpu:.2f} W"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=12, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = "results/codecarbon_dashboard.png"
    plt.savefig(output_path, dpi=120)
    print(f"[+] Advanced CodeCarbon Dashboard saved to {output_path}")

if __name__ == "__main__":
    create_codecarbon_dashboard()
