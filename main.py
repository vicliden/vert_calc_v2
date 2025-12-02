from jump_analysis import analyze_jumps_from_folder

results_2 = analyze_jumps_from_folder("hopp 2", save_plots=True, result_dir="hopp 2/results")

print("Results for 'hopp 2':")
for i, jump in enumerate(results_2, start=1):
    print(f"Jump {i}:")
    print(f"  takeoff_time_s: {jump['takeoff_time_s']}")
    print(f"  landing_time_s: {jump['landing_time_s']}")
    print(f"  airtime_s: {jump['airtime_s']}")
    print(f"  height_cm: {jump['height_cm']}")
    print(f"  height_inches: {jump['height_inches']}")

results_3 = analyze_jumps_from_folder(" hopp 3", save_plots=True, result_dir="hopp 3/results")

print("\nResults for hopp 3:")
for i, jump in enumerate(results_3, start=1):
    print(f"Jump {i}:")
    print(f"  takeoff_time_s: {jump['takeoff_time_s']}")
    print(f"  landing_time_s: {jump['landing_time_s']}")
    print(f"  airtime_s: {jump['airtime_s']}")
    print(f"  height_cm: {jump['height_cm']}")
    print(f"  height_inches: {jump['height_inches']}")

