import csv

input_file_name = 'datasets/processed/server_2026_03_12_gaps.csv'
output_file_name = 'datasets/differenced/server_2026_03_12_gaps.csv'

nom_freq = 5e6  # Nominal frequency for normalization (5 MHz)

clk_to_ch = {'AHM1': 2, 'AHM2': 10, 'CS1': 3, 'CSAC1': 1, 'CSAC2': 4, 'CSAC3': 5}
ch_to_clk = {v: k for k, v in clk_to_ch.items()}

reference_clk = 'AHM1'  # Use AHM1 as the reference channel for differencing

print(f'Reading data from {input_file_name}...')
with open(input_file_name, 'r') as infile:
    reader = csv.DictReader(infile)
    data = list(reader)
    channel_list = reader.fieldnames[1:]  # Get channel names from header, excluding timestamp

print(f'Calculating differences and writing to {output_file_name}...')
with open(output_file_name, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    # Write header for differenced data
    header = ['timestamp']
    for ch in channel_list:
        ch_num = int(ch[2:])  # Extract channel number from 'chX'
        if ch_num != clk_to_ch[reference_clk]:  # Skip reference channel
            header.append(f'{ch_to_clk[ch_num]} - {reference_clk}')
    writer.writerow(header)

    for row in data:
        timestamp = row['timestamp']
        reference_value = float(row[f'ch{clk_to_ch[reference_clk]}']) if row[f'ch{clk_to_ch[reference_clk]}'] else None
        
        differenced_row = [timestamp]
        for ch in channel_list:
            ch_num = int(ch[2:])
            if ch_num != clk_to_ch[reference_clk]:  # Skip reference channel
                current_value = float(row[ch]) if row[ch] else None
                if reference_value is not None and current_value is not None:
                    differenced_row.append((current_value - reference_value) / nom_freq)
                else:
                    differenced_row.append('')  # Append empty string if data is missing
                    print(f'Warning: Missing data for channel {ch} or reference channel at timestamp {timestamp}')
        
        writer.writerow(differenced_row)
