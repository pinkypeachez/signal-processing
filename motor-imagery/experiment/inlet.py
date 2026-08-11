# TODO: Threads fuer EEG und Markers Stream

from pylsl import StreamInfo, StreamInlet, resolve_byprop
import time
import csv
import os
import sys

'''Der Computer, auf dem dein Empfänger-Skript (das Inlet) läuft, 
wird automatisch zur Referenz!! Alle Zeitstempel von eingehenden Streams werden
 in die Zeitbasis dieses Empfänger-Computers umgerechnet'''

# Inlets öffnen
# To connect to a stream, a StreamInlet object must be created using the resolved StreamInfo

# EEG Stream
eeg_stream = resolve_byprop(prop="name", value="uni", timeout=2)
marker_stream = resolve_byprop (prop="name", value="MI_Markers", timeout=2)

if not eeg_stream: 
  print("Kein EEG Stream gefunden.")
  sys.exit()

if not marker_stream:
  print("Kein Marker Stream gefunden.")
  sys.exit()


print(eeg_stream, marker_stream)
eeg_inlet = StreamInlet(eeg_stream[0])
marker_inlet = StreamInlet(marker_stream[0])
marker_inlet.open_stream()

# Arrays zum Sammeln von Daten anlegen
eeg_data = []
marker_data = []

# ------------- Dateisystem anlegen
DIR_NAME = "data"

try:
    os.mkdir(DIR_NAME)
    print(f"Directory '{DIR_NAME}' created successfully.")
except FileExistsError:
    print(f"Directory '{DIR_NAME}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{DIR_NAME}'.")
except Exception as e:
    print(f"An error occurred: {e}")

# ------------- Outlet Daten sammeln

while True:
  try:
    # Marker
    sample_marker, ts = marker_inlet.pull_sample(timeout=0.0)  # 0 for non blocking
    if sample_marker is not None:
      #  print(f"MARKER: [{ts:.3f}] {sample_marker}")
        marker_data.append([ts, sample_marker])

    # EEG
    eeg_sample, ts_eeg = eeg_inlet.pull_chunk()
    if eeg_sample:
        for sample, ts in zip(eeg_sample, ts_eeg):
           # print(f"EEG: Sample {sample}")
            eeg_data.append([ts, sample])
           
    time.sleep(0.002)
  except KeyboardInterrupt:
   print("Inlet wurde manuell geschlossen")
   break
   

#print( "MARKER DATA: ", marker_data)
print("EEG data: ",eeg_data)

# ----- Die Daten aus Arrays in .csv laden und speichern
# newline=''> sonst schreibt csv-Writer zwischen jeder Zeile eine leere Zeile
eeg_csv = open('./data/eeg.csv', 'w', newline='')
with eeg_csv:
   writer = csv.writer(eeg_csv)

   for row in eeg_data:
      writer.writerow(row)

marker_csv = open('./data/marker.csv', 'w',newline='')
with marker_csv:
   writer = csv.writer(marker_csv)

   for row in marker_data:
      writer.writerow(row)

marker_inlet.close_stream()
eeg_inlet.close_stream()

