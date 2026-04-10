import numpy as np
import matplotlib.pyplot as plt

input_file_name = 'datasets/raw/server_2026_01_23.txt'

data = {}

print(f'Reading data from {input_file_name}...')
with open(input_file_name, 'r') as infile:

    for line in infile:
        parts = line.strip().split()
        if len(parts) == 6 and parts[0] == 'Channel':

            timestamp = int(parts[3])
            if timestamp not in data:
                data[timestamp] = {}

            channel = int(parts[1])
            meas = float(parts[5])

            data[timestamp][channel] = meas

# Convert data to numpy arrays for plotting
timestamps = sorted(data.keys())
channels = [1, 2, 3, 4, 5, 10]
channel_data = {ch: [] for ch in channels}
for timestamp in timestamps:
    for ch in channels:
        channel_data[ch].append(data[timestamp].get(ch, np.nan))  # Use NaN for missing data

# Plotting
plt.figure(figsize=(12, 8))
for ch in channels:
    # plt.scatter(timestamps[:-1], np.diff(channel_data[ch]), label=f'Channel {ch}')
    plt.scatter(timestamps, channel_data[ch], label=f'Channel {ch}')
plt.xlabel('Timestamp')
plt.ylabel('Measurement')
plt.title('Raw Measurements from Channels')
plt.legend()
plt.grid()
plt.show()