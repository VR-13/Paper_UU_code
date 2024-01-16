# this file reads openface output files and uses the AUs to extract the specific emotions from the faces in a rule-based method. 
# it outputs segments with emotions for each segment of 5 minutes (useful for comparing with annotations of segments). 


import csv
import os
import pandas as pd
import numpy as np



def translate_au_to_expression(au_activations):
    # Define rules for each facial expression based on AU activations
    expression_rules = {
        'neutral': not any(au_activations.values()),
        'happiness': au_activations['AU06c'] and au_activations['AU12c'],
        'sadness': au_activations['AU01c'] and au_activations['AU04c'] and au_activations['AU15c'],
        'surprise': au_activations['AU01c'] and au_activations['AU02c'] and au_activations['AU05c'] and au_activations['AU26c'],
        'fear': au_activations['AU01c'] and au_activations['AU02c'] and au_activations['AU04c'] and au_activations['AU05c'] and au_activations['AU07c'] and au_activations['AU20c'] and au_activations['AU26c'],
        'anger': au_activations['AU04c'] and au_activations['AU05c'] and au_activations['AU07c'] and au_activations['AU23c'],
        'disgust': au_activations['AU09c'] and au_activations['AU15c'] and au_activations['AU17c'],
        'contempt': au_activations['AU12r'] and au_activations['AU14r'],
        'pain': au_activations['AU04c'] and au_activations['AU06c'] and au_activations['AU07c'] and au_activations['AU09c'] and au_activations['AU10c'] and au_activations['AU12c'] and au_activations['AU20c'] and au_activations['AU25c'] and au_activations['AU26c'],
    }

    # Determine the expression based on the rules
    for expression, rule in expression_rules.items():
        if rule:
            return expression
    
    # Check if there's an expression that matches with one missing AU
    for expression, rule in expression_rules.items():
        missing_au = None
        for au, activated in au_activations.items():
            if not activated:
                missing_au = au
                break
        if missing_au is not None:
            modified_au_activations = au_activations.copy()
            modified_au_activations[missing_au] = True
            if rule and not any(modified_au_activations[au] for au in ['AU22c']):
                return expression
    
    # If no expression matches, return 'unknown'
    return 'unknown'


# Initialize dictionary to store expression counts
expression_counts = {
    'neutral': 0,
    'anger': 0,
    'disgust': 0,
    'fear': 0,
    'happiness': 0,
    'sadness': 0,
    'surprise': 0,
    'contempt': 0,
    'pain': 0,
    'unknown': 0,
}

# Frame rate of the video
frame_rate = 44.1

no_AUs = []

# Duration of each segment (in seconds)
segment_duration = 5 * 60  # 5 minutes

# Loop through files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv') and not filename.startswith('._'):
        file_path = os.path.join(folder_path, filename)

        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {filename}")
            continue

        # Read the input CSV file
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.split(' AU').str.join('_AU').str.strip()
        df.columns = df.columns.str.replace('_', '')

        # Check if 'AU01c' column exists
        if 'AU01c' not in df.columns:
            no_AUs.append(filename)
            continue

        # Group the data by face_id
        grouped_df = df.groupby('faceid')

        # Loop through face_id groups
        for face_id, group_data in grouped_df:
            if len(group_data) <= 3:
                continue  # Skip groups with less than or equal to 3 occurrences

            frame_column = group_data['frame']

            # Calculate the total number of frames
            total_frames = group_data.iloc[-1]['frame']

            # Determine the number of segments
            num_segments = int(total_frames / (frame_rate * segment_duration))

            # Initialize segment frame numbers and expression counts
            segment_frame_numbers = []
            segment_expression_counts = {expression: [] for expression in expression_counts}

            # Loop through segments
            for segment_index in range(num_segments):
                # Calculate start and end frame numbers of the segment
                # Find the closest frame indices for the start and end timestamps
                start_timestamp = pd.Timedelta(seconds=segment_index * segment_duration)
                end_timestamp = start_timestamp + pd.Timedelta(seconds=segment_duration)

                start_frame_index = np.abs(frame_column - start_timestamp.total_seconds() * frame_rate).idxmin()
                end_frame_index = np.abs(frame_column - end_timestamp.total_seconds() * frame_rate).idxmin()

                if end_frame_index == 0:
                    continue

                # Get the segment data
                segment_data = group_data.loc[start_frame_index:end_frame_index]

                # Get the segment frame number (e.g., start frame)
                segment_frame_number = segment_data['frame'].iloc[0]
                segment_frame_numbers.append(segment_frame_number)

                # Reset expression counts for each segment
                segment_expression_counts_segment = expression_counts.copy()

                # Loop through rows in the segment and calculate expression counts
                for row in range(len(segment_data)):
                    au_activations = {
                        'AU01c': bool(segment_data['AU01c'].iloc[row]),
                        'AU02c': bool(segment_data['AU02c'].iloc[row]),
                        'AU04c': bool(segment_data['AU04c'].iloc[row]),
                        'AU05c': bool(segment_data['AU05c'].iloc[row]),
                        'AU06c': bool(segment_data['AU06c'].iloc[row]),
                        'AU07c': bool(segment_data['AU07c'].iloc[row]),
                        'AU09c': bool(segment_data['AU09c'].iloc[row]),
                        'AU10c': bool(segment_data['AU10c'].iloc[row]),
                        'AU12c': bool(segment_data['AU12c'].iloc[row]),
                        'AU14c': bool(segment_data['AU14c'].iloc[row]),
                        'AU15c': bool(segment_data['AU15c'].iloc[row]),
                        'AU17c': bool(segment_data['AU17c'].iloc[row]),
                        'AU20c': bool(segment_data['AU20c'].iloc[row]),
                        'AU23c': bool(segment_data['AU23c'].iloc[row]),
                        'AU25c': bool(segment_data['AU25c'].iloc[row]),
                        'AU26c': bool(segment_data['AU26c'].iloc[row]),
                        'AU28c': bool(segment_data['AU28c'].iloc[row]),
                        'AU45c': bool(segment_data['AU45c'].iloc[row]),
                        'AU01r': int(segment_data['AU01r'].iloc[row]),
                        'AU02r': int(segment_data['AU02r'].iloc[row]),
                        'AU04r': int(segment_data['AU04r'].iloc[row]),
                        'AU05r': int(segment_data['AU05r'].iloc[row]),
                        'AU06r': int(segment_data['AU06r'].iloc[row]),
                        'AU07r': int(segment_data['AU07r'].iloc[row]),
                        'AU09r': int(segment_data['AU09r'].iloc[row]),
                        'AU10r': int(segment_data['AU10r'].iloc[row]),
                        'AU12r': int(segment_data['AU12r'].iloc[row]),
                        'AU14r': int(segment_data['AU14r'].iloc[row]),
                        'AU15r': int(segment_data['AU15r'].iloc[row]),
                        'AU17r': int(segment_data['AU17r'].iloc[row]),
                        'AU20r': int(segment_data['AU20r'].iloc[row]),
                        'AU23r': int(segment_data['AU23r'].iloc[row]),
                        'AU25r': int(segment_data['AU25r'].iloc[row]),
                        'AU26r': int(segment_data['AU26r'].iloc[row]),
                        'AU45r': int(segment_data['AU45r'].iloc[row]),
                    }

                    expression = translate_au_to_expression(au_activations)
                    segment_expression_counts_segment[expression] += 1

                # Append expression counts for the segment to the dictionary
                for expression, count in segment_expression_counts_segment.items():
                    segment_expression_counts[expression].append(count)

            # Create a dataframe for the segment expression counts
            segment_expression_df = pd.DataFrame(segment_expression_counts, index=segment_frame_numbers)

            # Save the dataframe to a CSV file with face_id in the filename
            output_file_path = os.path.join(output_csv_path, f"segment_expression_counts_face{face_id}_{filename}")
            segment_expression_df.to_csv(output_file_path)

            print(f"Segment expression counts for face_id {face_id} saved to: {output_file_path}")

# Print files with missing 'AU01c' column
print("Files with missing 'AU01c' column:")
for filename in no_AUs:
    print(filename)
