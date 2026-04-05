import csv

input_file_name = 'datasets/raw/server_2026_03_12.txt'
output_file_name = 'datasets/processed/server_2026_03_12.csv'
max_points = 6 * 7 * 60 * 60  # 6 channels * 7 hours * 60 minutes * 60 seconds

data = {}

print(f'Reading data from {input_file_name}...')
with open(input_file_name, 'r') as infile:

    i = 0
    for line in infile:
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

# Write the processed data to a CSV file
print(f'Writing processed data to {output_file_name}...')
with open(output_file_name, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['timestamp', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch10'])  # Write header

    for timestamp, channels in data.items():
        row = [timestamp]
        for ch in [1, 2, 3, 4, 5, 10]:
            row.append(channels[ch] if ch in channels else '')  # Append measurement or empty string if missing
            if ch not in channels:
                print(f'Warning: Missing data for channel {ch} at timestamp {timestamp}')
        writer.writerow(row)