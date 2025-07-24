from ingestion.bahrain_data_fetcher import fetch_multiple_years
from simulation.strategy import run_q_learning_strategy

df_laps, _ = fetch_multiple_years([2023])
driver = "1"  # Verstappen
driver_laps = df_laps[df_laps['DriverNumber'] == driver]

steps, total_time = run_q_learning_strategy(driver_laps, episodes=300)
print(f"\n📈 Strategy Steps for Driver #{driver}:")
for step in steps:
    print(step)

print(f"\n🕒 Total Simulated Race Time: {total_time:.2f} seconds")
