# this file extracts the sentiment (positive/negative) and specific emotions from transcriptions or text files
# the files have dutch text so are first translated into english as the model is trained on english text/corpus. 
# uses DTAI-KULeuven/robbert-v2-dutch-sentiment for sentiment and arpanghoshal/EmoRoBERTa for the specific 28 emotions. 
# links to models:
# - https://huggingface.co/DTAI-KULeuven/robbert-v2-dutch-sentiment
# - https://huggingface.co/arpanghoshal/EmoRoBERTa

import csv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import pipeline
import pandas as pd
import glob
import os
from collections import Counter
import traceback

model_name = "DTAI-KULeuven/robbert-v2-dutch-sentiment"
model = RobertaForSequenceClassification.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

def process_csv_files(csv_files, output_folder):
    for input_file in csv_files:
        try:
            print(input_file)
            # Path to the output CSV file
            output_file_segment = input_file.replace('.csv', '_segment.csv')
            # Open the input CSV file
            with open(input_file, 'r', newline='') as csv_input:
                reader = csv.DictReader(csv_input, delimiter=',')
                # Process each row in the input CSV file
                segment_data = []
                current_start_time = None
                current_end_time = None
                current_speaker = None
                emotion_counter = Counter()
                for row in reader:
                    timestamp = float(row['start']) * 1000
                    speaker = row['speaker']
                    dutch_transcript = row['text']
                    if current_start_time is None:
                        current_start_time = timestamp
                        current_end_time = timestamp
                        current_speaker = speaker
                    # Perform sentiment analysis for the current row
                    scores_sentiment = classifier(dutch_transcript)
                    emotions = emotion(dutch_transcript)
                    emotions_str = ', '.join([emotion['label'] for emotion in emotions])
                    # Update sentiment counter for the current 5-minute span
                    sentiment_label = scores_sentiment[0]['label']
                    emotion_counter.update(emotions_str.split(', '))
                    # Check if the current row is beyond the 5-minute span
                    if timestamp >= current_start_time + 5 * 60 * 1000:
                        # Create a dictionary for the segment data
                        segment_row = {
                            'Start Time': current_start_time,
                            'End Time': current_end_time,
                            'Speaker': current_speaker,
                            'Sentiment': sentiment_label,
                            **emotion_counter
                        }
                        segment_data.append(segment_row)
                        # Reset the variables for the next 5-minute span
                        current_start_time = timestamp
                        current_end_time = timestamp
                        current_speaker = speaker
                        emotion_counter = Counter()
                    else:
                        # Update the end time for the current 5-minute span
                        current_end_time = timestamp
                # Add the final segment data if any
                segment_row = {
                    'Start Time': current_start_time,
                    'End Time': current_end_time,
                    'Speaker': current_speaker,
                    'Sentiment': sentiment_label,
                    **emotion_counter
                }
                segment_data.append(segment_row)
            # Write the segment-level analysis to the output CSV file
            df_segment = pd.DataFrame(segment_data)
            df_segment.to_csv(os.path.join(output_folder, output_file_segment), index=False)
            # Print a message indicating the completion of the analysis for the current file
            print("Segment-level analysis completed for", input_file)
        except Exception as e:
            # Print the exception message
            print("Error processing file:", input_file)
            print("Exception:", str(e))
            traceback.print_exc()

path = "" # path to folder with transcription files 
output_folder = "" # path to folder where output files are stored


# List of CSV files to process
csv_files = glob.glob(os.path.join(path, '*.csv'))


os.makedirs(output_folder, exist_ok=True)

# Call the function to process the CSV files
process_csv_files(csv_files, output_folder)
