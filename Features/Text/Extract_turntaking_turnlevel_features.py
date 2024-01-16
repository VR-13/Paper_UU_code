# this code extract the turntaking and turnlevel features just like in the Bayerl et al (2022) paper - What can Speech and Language Tell us About the Working Alliance in Psychotherapy


# functions
import pandas as pd
import re
import itertools
import math
import librosa
from pyannote.audio import Pipeline
overlappipeline = Pipeline.from_pretrained("pyannote/overlapped-speech-detection",
                                    use_auth_token="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")

def convert_timestamp_to_seconds(timestamp):
    hours, minutes, seconds = timestamp.split(':')
    total_seconds = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    return total_seconds

def millisec(timeStr):
  # spl = timeStr.split(":")
  # s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  s = timeStr * 1000
  return s

def timestamp_to_seconds(timestamp):
    h, m, s = timestamp.split(':')
    seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return seconds

def timestamptoms(timeStr):
    spl = timeStr.split(":")
    seconds_str = spl[2]
    if "." in seconds_str:
        sec, msec = seconds_str.split(".")
        if len(msec) == 2:
            msec += "0"
    else:
        sec, msec = seconds_str, "000"
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + int(sec)) * 1000 + int(msec))
    return s

def speech_rate(text,speakerlist,startlist,endlist):
  # Initialize a dictionary to keep track of the speech duration for each speaker
  speech_durations = {}
  word_counts = {}
  turn = {}
  for speaker in speakerlist:
      word_counts[speaker] = 0

  # Loop over each row in the DataFrame
  for i in range(len(text)):
      # Get the speaker ID and the string of words for this segment
      speaker = speakerlist[i]
      words = text[i]
      start = millisec(startlist[i])
      end = millisec(endlist[i])

      # Count the number of words in this segment
      num_words = len(words.split())

      # Calculate the duration of this segment in seconds
      duration = (end - start) / 1000

      # Add the duration to the total speech duration for this speaker
      if speaker in speech_durations:
          speech_durations[speaker] += duration
          word_counts[speaker] += num_words
          turn[speaker] += 1

      else:
          speech_durations[speaker] = duration
          word_counts[speaker] += num_words
          turn[speaker] = 1

  # Calculate the speech rate per speaker
  speech_rates = {}
  for speaker in speech_durations.keys():
      duration = speech_durations[speaker]
      word_count = word_counts[speaker]
      speech_rate = word_count / (duration)
      speech_rates[speaker] = speech_rate

  durationpercentage_therapist = speech_durations['Therapist'] / (speech_durations['Therapist']+speech_durations['Patient']) *100
  durationpercentage_patient = speech_durations['Patient'] / (speech_durations['Therapist']+speech_durations['Patient']) *100
  turnlength_therapist = speech_durations['Therapist'] / turn['Therapist']
  turnlength_patient = speech_durations['Patient'] / turn['Patient']
  return turnlength_therapist,turnlength_patient,word_counts['Therapist'],word_counts['Patient'], speech_rates['Therapist'],speech_rates['Patient'],speech_durations['Therapist'],speech_durations['Patient'],durationpercentage_therapist,durationpercentage_patient


def percentageoverlap(nr_overlapturns,nr_of_turns):
  overlap = (nr_overlapturns / nr_of_turns) * 100
  return overlap

def participation_equality(speaker_durations):
    # speaker_durations is a list of total speech durations for each speaker
    # Calculate the average speech duration across all speakers
    avg_duration = sum(speaker_durations) / len(speaker_durations)
    # Calculate the maximum possible duration for a speaker
    max_duration = sum(speaker_durations)
    # Calculate the sum of squares of differences between each speaker's
    # speech duration and the average speech duration
    ssd = sum([(duration - avg_duration)**2 for duration in speaker_durations])
    # Calculate the participation equality score using the formula
    peq = 1 - (ssd / max_duration)
    return peq


def turn_level_freedom(turnforeachspeaker):
    # turnforeachspeaker is a list of tuples, where each tuple contains the name of the speaker
    # and the duration of their turn
    # Extract the list of speakers from the turnforeachspeaker data
    speakers = list(set([turn[0] for turn in turnforeachspeaker]))
    # Initialize variables for calculating the conditional entropy and maximum conditional entropy
    cond_ent_sum = 0
    max_cond_ent_sum = 0
    num_turns = len(turnforeachspeaker)
    # Loop over all pairs of adjacent turnforeachspeaker
    for i in range(num_turns - 1):
        x = turnforeachspeaker[i][0]
        y = turnforeachspeaker[i+1][0]
        # Calculate the number of times speaker y follows speaker x
        num_follows = sum([turn[0] == y for turn in turnforeachspeaker[i+1:]])
        # Calculate the probability of speaker y following speaker x
        prob_follows = num_follows / (num_turns - i - 1)
        # Calculate the conditional entropy for this pair of speakers
        if prob_follows > 0:
            cond_ent = -prob_follows * math.log2(prob_follows)
            cond_ent_sum += cond_ent

        # Calculate the maximum possible conditional entropy for this pair of speakers
        max_cond_ent = -0.5 * math.log2(prob_follows)
        max_cond_ent_sum += max_cond_ent

    # Calculate the turn-level freedom score using the formula
    fcond = 1 - (cond_ent_sum / max_cond_ent_sum)
    return fcond


def runfeat(filename, output_path, video_nr, video_path, first, final_df, pause_df):
  # read in the csv file with data
  csv_df = pd.read_csv(filename,na_values=[''])
  text = csv_df['text']
  startlist = csv_df['start']
  endlist = csv_df['end']

  speaker_list = []
  turns = []
  # change speaker id to label
  for i in range(len(csv_df)):
    if csv_df['speaker'][i] == 'SPEAKER_01':
      speaker_list.append('Therapist')
    elif csv_df['speaker'][i] == 'SPEAKER_00':
      speaker_list.append('Patient')
    else:
      speaker_list.append('None')

    # create turn duration list
    start = millisec(csv_df['start'][i])
    end = millisec(csv_df['end'][i])
    turns.append([start,end])

  # the number of turns for each speaker
  nr_of_turns = len(csv_df)
  speaker_therapist = [idx for idx, element in enumerate(speaker_list) if element == "Therapist"]
  speaker_patient = [idx for idx, element in enumerate(speaker_list) if element == "Patient"]
  nr_speaker_therapist = len(speaker_therapist)
  nr_speaker_patient = len(speaker_patient)

  # select speaking turns per speaker and get duration per turn
  duration_patient = 0
  duration_therapist = 0
  duration_na = 0
  turnforeachspeaker = []

  for g in range(len(csv_df)):
    duration = (turns[g][1]) - (turns[g][0])
    if csv_df['speaker'][g] == 'SPEAKER_01':
      # duration for speaker 0
      duration_therapist += duration
      turntemp = ("Therapist",duration/1000)
    elif csv_df['speaker'][g] == 'SPEAKER_00':
      # duration for speaker 1
      duration_patient += duration
      turntemp = ("Patient",duration/1000)
    else:
      duration_na += 1
      turntemp = ("Na",duration/1000)

    turnforeachspeaker.append(turntemp)

  # average duration
  duration_therapist = (duration_therapist / nr_speaker_therapist) / 1000 # in millisec
  duration_patient = (duration_patient / nr_speaker_patient) / 1000 # in millisec

  # participation equality
  peq = participation_equality([duration_therapist,duration_patient])
  # turn level freedom
  if len(turnforeachspeaker) > 0:
    fcond = turn_level_freedom(turnforeachspeaker)
  else:
    fcond = 0

  # overlap
  overlapsegments = overlappipeline(video_path)

  # number of overlapping turns
  nr_overlapturns = len(overlapsegments)
  overlap = percentageoverlap(nr_overlapturns,nr_of_turns)

  # calculate various features
  turnlength_therapist,turnlength_patient, wordcount_speaker_therapist,wordcount_speaker_patient, speechrate_speaker_therapist,speechrate_speaker_patient,speechduration_therapist,speechduration_patient,durationpercentage_therapist,durationpercentage_patient = speech_rate(text,speaker_list,startlist,endlist)

  silencesegment = []
  silenceduration= []
  # loop through speech segments and if there is a silence between the segments, determine length
  for i in range(len(turns)-1):
    # silence segment
    durationsilence = turns[i+1][0] - turns[i][1]
    silenceduration.append(durationsilence)
    silencesegment.append([turns[i][1],turns[i+1][0]])

  pause_segment = []
  pause_duration = []
  pause_type = []
  pause_speaker = []
  speechsegments = turns
  # loop through every silence segment
  for i in range(len(silencesegment)):
    for j in range(len(speechsegments)):
      # check if silence segment falls within speech segment
      # the added .001 is to prevent the system from skipping silences based on small differences
      if silencesegment[i][0] >= speechsegments[j][0] - 0.001 and silencesegment[i][1] <= speechsegments[j][1] + 0.001:
        # silence segment falls within this speech segment
        # add variables
        pause_segment.append(silencesegment[i])
        pause_duration.append(silenceduration[j])
        pause_type.append('within')
        pause_speaker.append(speaker_list[j])
        # go to next silence segment
        break

      # check if silence segment falls between two speakers
      elif silencesegment[i][0] >= speechsegments[j][1] - 0.001 and silencesegment[i][1] <= speechsegments[j+1][0] +0.001 and j < len(speechsegments):
        # silence segment falls between two speech segments
        # add variables
        pause_segment.append(silencesegment[i])
        pause_duration.append(silenceduration[j])
        pause_type.append('between')
        pause_speaker.append([speaker_list[j],speaker_list[j+1]])
        # go to next silence segment
        break
      # else, loop to next speech segment to see if it falls within/between that one

  # save in csv
  final_df.loc[len(final_df)] = [video_nr, nr_of_turns, nr_speaker_therapist, nr_speaker_patient, duration_therapist, duration_patient, peq, fcond, turnlength_therapist, turnlength_patient, wordcount_speaker_therapist, wordcount_speaker_patient, speechrate_speaker_therapist, speechrate_speaker_patient, speechduration_therapist, speechduration_patient, durationpercentage_therapist, durationpercentage_patient, overlap]

  # append a row to the pause_df DataFrame
  pause_df.loc[len(pause_df)] = [video_nr, pause_segment, pause_duration, pause_type, pause_speaker]

  return final_df, pause_df, duration_na


# main part of code
import glob
import os
import traceback
import pandas as pd
path="" # path to folder with transcription files
output_path = "" # path to output folder

# create an empty DataFrame with column names
columns = ['video_id','nr_of_turns', 'turns_therapist', 'turns_speaker_patient', 'duration_therapist', 'duration_patient', 'participation_equality', 'turn_level_freedom', 'avgturnlength_therapist', 'avgturnlength_patient', 'wordcount_speaker_therapist', 'wordcount_speaker_patient', 'speechrate_speaker_therapist', 'speechrate_speaker_patient', 'speechduration_therapist', 'speechduration_patient', 'durationpercentage_therapist', 'durationpercentage_patient', 'overlap']
final_df = pd.DataFrame(columns=columns)

# create an empty DataFrame for pause features
pause_columns = ['video_id','pause_segment', 'pause_duration', 'pause_type', 'pause_speaker']
pause_df = pd.DataFrame(columns=pause_columns)

csvfiles = glob.glob(os.path.join(path,'*.csv'))
for nr in range(len(csvfiles)):
  id = str(csvfiles[nr].split('/')[5].replace("wav.csv", "").replace("segments", ""))
  video_path = csvfiles[nr].replace(".csv", "").replace(".wav", ".WAV").replace("segments", "")
  print(video_path)
  print(id)
  print("running file nr: ",nr)
  try:
    final_df, pause_df, duration_na = runfeat(csvfiles[nr], output_path, id, video_path, nr, final_df, pause_df) #filename, output_path, video_nr, video_path, first
  except Exception as e:
            # Print the exception message
    print("Error processing file:", id)
    print("Exception:", str(e))
    traceback.print_exc()


final_df.to_csv(output_path + 'allfeatures' + id + '.csv', index=False)
pause_df.to_csv(output_path + 'allpausefeatures' + id + '.csv', index=False)
