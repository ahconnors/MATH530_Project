import csv
import numpy as np

input_file_name = 'datasets/raw/server_2026_03_12.txt'
output_file_name = 'datasets/processed/server_2026_03_12_gaps.csv'
# burn_in_time = 6 * 48 * 60 * 60  # 6 channels * 1 day in seconds
burn_in_time = 0
max_points = 6 * 24 * 60 * 60  # 6 channels * 24 hours * 60 minutes * 60 seconds

data = {}

print(f'Reading data from {input_file_name}...')
with open(input_file_name, 'r') as infile:
    lines = infile.readlines()

    i = 0
    for line in lines[burn_in_time:]:  # Skip burn-in period
        parts = line.strip().split()
        if len(parts) == 6 and parts[0] == 'Channel':

            timestamp = int(parts[3])
            if timestamp not in data:
                data[timestamp] = {}

            channel = int(parts[1])
            meas = float(parts[5])

            data[timestamp][channel] = meas
        
            i += 1
            if i % (max_points // 100) == 0:
                print(f'{i // (max_points // 100)}% of points processed...')
            if i >= max_points:
                print('Reached maximum number of points to process.')
                break

min_timestamp = min(data.keys())
max_timestamp = max(data.keys())
time_range = np.arange(min_timestamp, max_timestamp + 1)

# Write the processed data to a CSV file
print(f'Writing processed data to {output_file_name}...')
with open(output_file_name, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['timestamp', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch10'])  # Write header

    for timestamp in time_range:
        row = [timestamp]
        if timestamp not in data:
            print(f'Warning: Missing data for timestamp {timestamp}')
            row.extend([''] * 6)  # Append empty strings for all channels
        else:
            for ch in [1, 2, 3, 4, 5, 10]:
                row.append(data[timestamp].get(ch, ''))  # Append measurement or empty string if missing
        writer.writerow(row)