# this code is similar to the other emotion extraction from openfaceoutput but this counts the emotions per file and thus does not segment data.

import csv
import os
import pandas as pd
import numpy as np

# Path to input CSV file
folder_path = '/Volumes/My Passport/openface/threshold/'

# Path to output CSV file
output_csv_path = '/Volumes/My Passport/openface/threshold/output/'

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

        # Initialize expression counts for each face_id
        face_id_expression_counts = {}

        # Loop through face_id groups
        for face_id, group_data in grouped_df:
            if len(group_data) <= 3:
                continue  # Skip groups with less than or equal to 3 occurrences

            # Reset expression counts for each face_id
            expression_counts_face_id = expression_counts.copy()

            # Loop through rows and calculate expression counts
            for row in range(len(group_data)):
                au_activations = {
                    'AU01c': bool(group_data['AU01c'].iloc[row]),
                    'AU02c': bool(group_data['AU02c'].iloc[row]),
                    'AU04c': bool(group_data['AU04c'].iloc[row]),
                    'AU05c': bool(group_data['AU05c'].iloc[row]),
                    'AU06c': bool(group_data['AU06c'].iloc[row]),
                    'AU07c': bool(group_data['AU07c'].iloc[row]),
                    'AU09c': bool(group_data['AU09c'].iloc[row]),
                    'AU10c': bool(group_data['AU10c'].iloc[row]),
                    'AU12c': bool(group_data['AU12c'].iloc[row]),
                    'AU14c': bool(group_data['AU14c'].iloc[row]),
                    'AU15c': bool(group_data['AU15c'].iloc[row]),
                    'AU17c': bool(group_data['AU17c'].iloc[row]),
                    'AU20c': bool(group_data['AU20c'].iloc[row]),
                    'AU23c': bool(group_data['AU23c'].iloc[row]),
                    'AU25c': bool(group_data['AU25c'].iloc[row]),
                    'AU26c': bool(group_data['AU26c'].iloc[row]),
                    'AU28c': bool(group_data['AU28c'].iloc[row]),
                    'AU45c': bool(group_data['AU45c'].iloc[row]),
                    'AU01r': int(group_data['AU01r'].iloc[row]),
                    'AU02r': int(group_data['AU02r'].iloc[row]),
                    'AU04r': int(group_data['AU04r'].iloc[row]),
                    'AU05r': int(group_data['AU05r'].iloc[row]),
                    'AU06r': int(group_data['AU06r'].iloc[row]),
                    'AU07r': int(group_data['AU07r'].iloc[row]),
                    'AU09r': int(group_data['AU09r'].iloc[row]),
                    'AU10r': int(group_data['AU10r'].iloc[row]),
                    'AU12r': int(group_data['AU12r'].iloc[row]),
                    'AU14r': int(group_data['AU14r'].iloc[row]),
                    'AU15r': int(group_data['AU15r'].iloc[row]),
                    'AU17r': int(group_data['AU17r'].iloc[row]),
                    'AU20r': int(group_data['AU20r'].iloc[row]),
                    'AU23r': int(group_data['AU23r'].iloc[row]),
                    'AU25r': int(group_data['AU25r'].iloc[row]),
                    'AU26r': int(group_data['AU26r'].iloc[row]),
                    'AU45r': int(group_data['AU45r'].iloc[row]),
                }

                expression = translate_au_to_expression(au_activations)
                expression_counts_face_id[expression] += 1

            # Store expression counts for the face_id
            face_id_expression_counts[face_id] = expression_counts_face_id

        # Create a dataframe for the expression counts
        expression_counts_df = pd.DataFrame.from_dict(face_id_expression_counts, orient='index')

        # Save the dataframe to a CSV file with face_id rows
        output_file_path = os.path.join(output_csv_path, f"expression_counts_{filename}")
        expression_counts_df.to_csv(output_file_path)

        print(f"Expression counts for {filename} saved to: {output_file_path}")

# Print files with missing 'AU01c' column
print("Files with missing 'AU01c' column:")
for filename in no_AUs:
    print(filename)
