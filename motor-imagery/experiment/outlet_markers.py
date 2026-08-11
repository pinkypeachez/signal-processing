# Anzeige: -->
# Marker fuer die Klasse wird in LSL Stream geschrieben (STREAM OUTLET)
# 0: Links / 1: Rechts / 2: Ruhe 

from psychopy import visual, core, event
import numpy as np
from pylsl import StreamInfo, StreamOutlet, IRREGULAR_RATE
import random
import sys


# ---------- LSL Outlet ----------
markers_stream_info = StreamInfo(
    name="MI_Markers",
    type="Markers",
    channel_count=1,
    nominal_srate=IRREGULAR_RATE,
    channel_format='string',
    source_id="psychopy_marker_stream"
)
outlet = StreamOutlet(markers_stream_info)
print("LSL Marker-Stream wurde erstellt.")

print("Warte auf Verbindung mit Inlet...")
outlet.wait_for_consumers(timeout=30)
if outlet.have_consumers():
    print("Verbunden. Experiment wird gestartet..")
else: 
    print("Keine Verbindung zum Inlet. " \
    "Experiment wird nicht durchgefuert")
    sys.exit()

# 2 Runs mit je 20 Trials (10 links / 10 rechts)
# 1 Minute Pause

# --------- Parameter
BLOCKS = 1
TRIALS_PER_CLASS = 5      # Trials per Klass per Block!
FIXATE_DUR = 2.0
CUE_DUR = 1.25            # Pfeil
MI_DUR = 6.0             # Motor Imagery Phase


BLOCK_PAUSE = 60 #60 sekunden

# ITI Inter Trial Interval
#iti = np.random.choice(np.arange(1.5, 2.5, step=0.25),
#                       size=10 )

FULLSCREEN = True
BG_COLOR = [-1, -1, -1]
TXT_COLOR = [1, 1, 1]
KREUZ_GRAU = [0, 0, 0]

# ---------- Fenster & Timing-Setup ----------
win = visual.Window(fullscr=FULLSCREEN, color=BG_COLOR, units="pix")

# Ein Timer für alle zeitbasierten Phasen
trial_timer = core.Clock()

# ---------- Stimuli ----------
# = visual.TextStim(win, text="", color=TXT_COLOR, height=120)
black_background = visual.TextStim(win, text="", color=BG_COLOR, height=120)
fix_grau = visual.TextStim(win, text="+", color=KREUZ_GRAU, height=120)
arrow_left = visual.TextStim(win, text="➜", color=TXT_COLOR, height=120,  flipHoriz=True)
arrow_right = visual.TextStim(win, text="➜", color=TXT_COLOR, height=120)
block_pause = visual.TextStim(win, text="Pause", color=KREUZ_GRAU, height=120)
kb_abort_keys = ["escape"]


# -------------- Trial Liste erstellen
# 0: Links / 1: Rechts / 2: Ruhe
trial_list = np.array([0,1], dtype=np.int8)
trial_list = np.repeat(trial_list, [TRIALS_PER_CLASS, TRIALS_PER_CLASS])
#np.random.shuffle(trial_list)



# ---------- Trial-Loop ----------
for block in range(BLOCKS):
 print(f"Starte Block {block}")
 np.random.shuffle(trial_list)

 for t, cue in enumerate(trial_list):
    #-------- Fixationskreuz
    fix_grau.draw()
    win.callOnFlip(outlet.push_sample, ["START_FIXATE"])
    win.flip()
    trial_timer.reset()
    while trial_timer.getTime() < FIXATE_DUR:
        if event.getKeys(kb_abort_keys):
            win.close()
            core.quit()
        fix_grau.draw()
        win.flip()

    # -------------------------------    Cue-Phase   --------------------------------------
    pfeil = arrow_left if cue == 0 else arrow_right
    marker = ["LINKS"] if cue == 0 else ["RECHTS"]
    print(marker) #####################################

    pfeil.draw() #--------------------------------- HIER IST WAS GEMALT WIRD
    win.callOnFlip(outlet.push_sample, marker)
    win.flip()
    trial_timer.reset()
    while trial_timer.getTime() < CUE_DUR:
        if event.getKeys(kb_abort_keys):
            win.close()
            core.quit()
        pfeil.draw()
        win.flip()

    # --- Motor Imagery (MI) Phase ---
    fix_grau.draw()
    win.callOnFlip(outlet.push_sample, ["MI_START"])
    print("MI_START") #####################################
    win.flip()
    trial_timer.reset() # Timer starten
    while trial_timer.getTime() < MI_DUR:
        # Fixationskreuz wird auf dem Bildschirm gehalten
        fix_grau.draw() 
        win.flip()
        if event.getKeys(kb_abort_keys):
            win.close()
            core.quit()
    
    # --- ITI: Inter-Trial-Interval ---
    black_background.draw()
    win.callOnFlip(outlet.push_sample, ["MI_END_REST_START"])
    win.flip()
    trial_timer.reset() # Timer starten
    while trial_timer.getTime() < np.random.choice(np.arange(1.5, 2.5, step=0.25)):
        black_background.draw()
        win.flip()
        if event.getKeys(kb_abort_keys):
            win.close()
            core.quit()

  # --- PAUSE ZWISCHEN BLOCKS (1 Minute) ---
 if block != 0:
    block_pause.draw()
    win.callOnFlip(outlet.push_sample, ["BLOCK_PAUSE"])
    win.flip()
    trial_timer.reset() # Timer starten
    while trial_timer.getTime() < BLOCK_PAUSE:
          block_pause.draw()
          win.flip()
          if event.getKeys(kb_abort_keys):
              win.close()
              core.quit()

# ---------- Aufräumen ----------
print("Experiment beendet.")
win.close()
core.quit()