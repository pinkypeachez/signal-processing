"""
STDP-Mustererkennung: Golden Model (Software-Referenz vor der Hardware-Implementierung)
=========================================================================================

Aufgabe (angelehnt an Masquelier, Guyonneau & Thorpe 2008):
Ein einzelnes LIF-Neuron mit STDP soll lernen, ein wiederkehrendes, fest
eingefrorenes raeumlich-zeitliches Spike-Muster zu erkennen, das statistisch
nicht von Hintergrundrauschen zu unterscheiden ist (gleiche mittlere Rate,
nur die exakte zeitliche Feinstruktur wiederholt sich).

Dieses Skript ist bewusst so geschrieben, dass jeder Baustein (Leaky-Integrator,
Trace, Schwellwert-Adaption) 1:1 als spaeterer SystemVerilog-Block wiedererkennbar
ist -- diskrete Zeitschritte, einfache exponentielle Zerfaelle, keine "exotischen"
numpy-Tricks.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1) Parameter
# ---------------------------------------------------------------------------

SEED = 42

# --- Datensatz ---
N_CHANNELS = 16                 # Anzahl afferenter Eingangskanaele
DT_MS = 1.0                     # Simulationsschritt [ms]
DURATION_MS = 60_000            # Gesamtlaenge der Simulation [ms] (60 s)
BACKGROUND_RATE_HZ = 10.0       # Poisson-Hintergrundrate pro Kanal
PATTERN_DURATION_MS = 50.0      # Laenge des eingebetteten Musters
CYCLE_MS = 300.0                # mittlerer Abstand zwischen moeglichen Musterfenstern
PATTERN_PROB = 0.6              # Wahrscheinlichkeit, dass das Muster in einem Zyklus vorkommt
COINCIDENCE_CHANNELS = 4        # Anzahl Kanaele, die im Muster einmal synchron feuern
                                 # (bewusst so kalibriert, dass diese Koinzidenz ALLEIN mit den
                                 # Start-Gewichten NICHT zuverlaessig ueber Schwelle kommt --
                                 # sonst wuerde das Neuron "erkennen", ohne je zu lernen)

# --- LIF-Neuron ---
TAU_MEM_MS = 10.0               # Membran-Zeitkonstante
V_RESET = 0.0
THETA_BASELINE = 0.4            # Basis-Schwellwert
THETA_PLUS = 0.5                # Schwellwert-Zuwachs pro Ausgangs-Spike (Homoeostase)
TAU_THETA_MS = 100.0            # Zeitkonstante des Schwellwert-Abklingens
REFRACTORY_MS = 5.0

# --- STDP (trace-basiert) ---
TAU_PRE_MS = 20.0
TAU_POST_MS = 20.0
A_PLUS = 0.010                  # LTP-Schrittweite (bei Post-Spike)
A_MINUS = 0.011                 # LTD-Schrittweite (bei Pre-Spike) -- etwas groesser als A_PLUS
                                 # fuer stabile Gewichte (Standard-Asymmetrie)
W_MIN, W_MAX = 0.0, 1.0
W_INIT_MEAN = 0.07
W_INIT_STD = 0.014
# Weiche (multiplikative) Grenzen statt hartem Clipping: LTP wird schwaecher,
# je naeher w an W_MAX ist, LTD wird schwaecher, je naeher w an W_MIN ist.
# Das ist der Standard-Fix gegen die Sättigungs-Runaway-Dynamik von additivem STDP.
SOFT_BOUNDS = True

# Synaptische Normalisierung: die Summe aller Gewichte eines Neurons wird periodisch
# auf einen festen Wert zurueckgesetzt. Ohne das driften bei trace-basiertem STDP
# fast alle Gewichte gemeinsam nach oben (Potenzierung trifft bei jedem Output-Spike
# alle Kanaele mit nicht-leerer Trace, Depression nur die gerade aktiven -> Bias nach oben).
# Normalisierung erzwingt Konkurrenz zwischen den Synapsen (wer waechst, nimmt anderen
# etwas weg) -- Standardtechnik in unueberwachten STDP-Netzen (u.a. bei Diehl & Cook).
NORMALIZE_EVERY_STEPS = 50


# ---------------------------------------------------------------------------
# 2) Datensatz-Generierung
# ---------------------------------------------------------------------------

def generate_dataset(rng):
    """Erzeugt eine (N_CHANNELS x n_steps) boolesche Spike-Matrix mit eingebettetem,
    exakt wiederkehrendem Muster. Gibt außerdem die Liste der Muster-Onset-Zeitschritte
    zurueck (Ground Truth)."""

    n_steps = int(DURATION_MS / DT_MS)
    p_spike = BACKGROUND_RATE_HZ * (DT_MS / 1000.0)

    # Hintergrund: unabhaengiges Poisson-Rauschen auf allen Kanaelen
    spikes = rng.random((N_CHANNELS, n_steps)) < p_spike

    # Ein festes Muster-Template erzeugen (gleiche Grund-Statistik wie Hintergrund,
    # aber ab jetzt eingefroren -> bei jeder Wiederholung bit-identisch)
    n_pattern_steps = int(PATTERN_DURATION_MS / DT_MS)
    template = rng.random((N_CHANNELS, n_pattern_steps)) < p_spike

    # Zusaetzlich: ein kleines synchrones "Landmark"-Ereignis einbauen -- COINCIDENCE_CHANNELS
    # Kanaele feuern innerhalb weniger ms exakt gleichzeitig. Das gibt dem Integrator
    # eine echte, gut detektierbare Koinzidenz, an der STDP ansetzen kann, statt sich
    # rein auf die (bei wenigen Kanaelen und moderater Wiederholungszahl sehr schwache)
    # Korrelation aus rein unabhaengigem, wiederholtem Rauschen verlassen zu muessen.
    landmark_step = n_pattern_steps // 2
    coincidence_channels = rng.choice(N_CHANNELS, size=COINCIDENCE_CHANNELS, replace=False)
    template[:, landmark_step] = False
    template[coincidence_channels, landmark_step] = True

    # Moegliche Zyklen durchgehen und mit PATTERN_PROB das Muster einfuegen
    n_cycles = int(DURATION_MS // CYCLE_MS)
    max_jitter = CYCLE_MS - PATTERN_DURATION_MS - 20.0  # Sicherheitsabstand, kein Overlap
    pattern_onsets = []

    for k in range(n_cycles):
        if rng.random() < PATTERN_PROB:
            cycle_start = k * CYCLE_MS
            jitter = rng.uniform(0.0, max_jitter)
            onset_ms = cycle_start + jitter
            onset_step = int(onset_ms / DT_MS)
            spikes[:, onset_step:onset_step + n_pattern_steps] = template
            pattern_onsets.append(onset_step)

    return spikes, np.array(pattern_onsets), template


# ---------------------------------------------------------------------------
# 3) LIF-Neuron + trace-basiertes STDP (Online-Simulation, Schritt fuer Schritt)
# ---------------------------------------------------------------------------

def simulate_lif_stdp(spikes, rng):
    n_channels, n_steps = spikes.shape

    decay_mem = np.exp(-DT_MS / TAU_MEM_MS)
    decay_theta = np.exp(-DT_MS / TAU_THETA_MS)
    decay_pre = np.exp(-DT_MS / TAU_PRE_MS)
    decay_post = np.exp(-DT_MS / TAU_POST_MS)
    refractory_steps = int(REFRACTORY_MS / DT_MS)

    w = rng.normal(W_INIT_MEAN, W_INIT_STD, size=n_channels).clip(W_MIN, W_MAX)
    target_weight_sum = w.sum()   # fester "Budget"-Wert, auf den periodisch normalisiert wird
    V = 0.0
    theta = THETA_BASELINE
    x = np.zeros(n_channels)   # Pre-Traces (eine pro Kanal)
    y = 0.0                    # Post-Trace (ein Ausgangsneuron)
    refractory_countdown = 0

    output_spike_steps = []
    # Log fuer Plots: Gewichte periodisch mitschreiben
    log_every = int(1000 / DT_MS)  # alle 1000 ms
    weight_log_steps = []
    weight_log = []

    for t in range(n_steps):
        active = spikes[:, t]          # welche Kanaele feuern in diesem Schritt

        # --- Membran-Update ---
        if refractory_countdown > 0:
            V = V_RESET
            refractory_countdown -= 1
        else:
            I_t = np.sum(w[active])
            V = V * decay_mem + I_t

        # --- Pre-Traces aktualisieren (Zerfall, dann bei Spike auf 1 setzen) ---
        x *= decay_pre
        x[active] = 1.0

        # --- Depression: fuer jeden gerade aktiven Kanal proportional zur Post-Trace ---
        if np.any(active):
            if SOFT_BOUNDS:
                w[active] -= A_MINUS * y * (w[active] - W_MIN)
            else:
                w[active] -= A_MINUS * y
            np.clip(w, W_MIN, W_MAX, out=w)

        # --- Post-Trace zerfaellt ---
        y *= decay_post

        # --- Schwellwert klingt Richtung Baseline ab ---
        theta = THETA_BASELINE + (theta - THETA_BASELINE) * decay_theta

        # --- Spike-Bedingung pruefen ---
        if refractory_countdown == 0 and V >= theta:
            output_spike_steps.append(t)
            V = V_RESET
            refractory_countdown = refractory_steps
            theta += THETA_PLUS            # Homoeostase: Schwelle steigt
            y = 1.0                        # Post-Trace setzen

            # Potenzierung: alle Kanaele profitieren proportional zu ihrer Pre-Trace
            if SOFT_BOUNDS:
                w += A_PLUS * x * (W_MAX - w)
            else:
                w += A_PLUS * x
            np.clip(w, W_MIN, W_MAX, out=w)

        if t % NORMALIZE_EVERY_STEPS == 0:
            current_sum = w.sum()
            if current_sum > 1e-6:
                w *= (target_weight_sum / current_sum)

        if t % log_every == 0:
            weight_log_steps.append(t)
            weight_log.append(w.copy())

    return {
        "output_spike_steps": np.array(output_spike_steps),
        "final_weights": w,
        "weight_log_steps": np.array(weight_log_steps) * DT_MS,
        "weight_log": np.array(weight_log),
    }


# ---------------------------------------------------------------------------
# 4) Auswertung: hat das Neuron gelernt, das Muster zu erkennen?
# ---------------------------------------------------------------------------

def offsets_to_nearest_onset(spike_steps, onset_steps, window_ms=150.0):
    """Fuer jeden Output-Spike: zeitlicher Abstand zum naechstgelegenen
    VORHERGEHENDEN Muster-Onset, begrenzt auf ein Fenster (in ms)."""
    offsets = []
    if len(onset_steps) == 0 or len(spike_steps) == 0:
        return np.array(offsets)
    onset_steps_sorted = np.sort(onset_steps)
    for s in spike_steps:
        idx = np.searchsorted(onset_steps_sorted, s, side="right") - 1
        if idx >= 0:
            offset_ms = (s - onset_steps_sorted[idx]) * DT_MS
            if 0 <= offset_ms <= window_ms:
                offsets.append(offset_ms)
    return np.array(offsets)


def main():
    rng = np.random.default_rng(SEED)

    spikes, pattern_onsets, template = generate_dataset(rng)
    result = simulate_lif_stdp(spikes, rng)

    out_steps = result["output_spike_steps"]
    n_steps = spikes.shape[1]

    print(f"Gesamtdauer: {DURATION_MS/1000:.0f} s, Muster-Vorkommen: {len(pattern_onsets)}")
    print(f"Anzahl Ausgangs-Spikes: {len(out_steps)}")

    # Die eigentlich aussagekraeftige Metrik: liegt ein Output-Spike praezise im
    # 15-35ms-Fenster nach einem Onset (dort, wo das synchrone Landmark-Ereignis sitzt)?
    # -> das ist "Wiedererkennung", nicht nur "irgendwo im 150ms-Fenster gefeuert".
    def recall_in_landmark_window(spike_steps, onset_steps):
        offs = offsets_to_nearest_onset(spike_steps, onset_steps)
        hit_onsets = 0
        offs_sorted_idx = np.where((offs >= 15) & (offs < 35))[0]
        return len(offs_sorted_idx), len(onset_steps)

    n_windows = 6
    window_steps = n_steps // n_windows
    print("Erkennungsrate (Treffer im 15-35ms-Landmark-Fenster) je 10s-Abschnitt:")
    for i in range(n_windows):
        lo, hi = i * window_steps, (i + 1) * window_steps
        seg_spikes = out_steps[(out_steps >= lo) & (out_steps < hi)]
        seg_onsets = pattern_onsets[(pattern_onsets >= lo) & (pattern_onsets < hi)]
        hits, total = recall_in_landmark_window(seg_spikes, seg_onsets)
        print(f"  {lo*DT_MS/1000:5.0f}-{hi*DT_MS/1000:5.0f} s: {hits:3d} / {total:3d} Muster erkannt")

    half = n_steps // 2
    early_offsets = offsets_to_nearest_onset(out_steps[out_steps < half], pattern_onsets[pattern_onsets < half])
    late_offsets = offsets_to_nearest_onset(out_steps[out_steps >= half], pattern_onsets[pattern_onsets >= half])

    # ------------------------------------------------------------------
    # Plot 1: Beispielausschnitt des Eingangsdatensatzes mit Muster-Fenstern
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    zoom_ms = 1800
    zoom_steps = int(zoom_ms / DT_MS)
    for ch in range(N_CHANNELS):
        ts = np.where(spikes[ch, :zoom_steps])[0] * DT_MS
        ax.scatter(ts, np.full_like(ts, ch), s=6, color="black")
    n_pattern_steps = int(PATTERN_DURATION_MS / DT_MS)
    for onset in pattern_onsets:
        if onset * DT_MS < zoom_ms:
            ax.axvspan(onset * DT_MS, (onset + n_pattern_steps) * DT_MS,
                       color="orange", alpha=0.25)
    ax.set_xlabel("Zeit [ms]")
    ax.set_ylabel("Kanal")
    ax.set_title("Eingangsdatensatz (Ausschnitt) -- orange = eingebettetes Muster")
    fig.tight_layout()
    fig.savefig("./input_raster_pattern_example.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: Lernergebnis (3 Teilplots)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # (a) Output-Raster ueber die gesamte Simulation + Onsets
    ax = axes[0]
    ax.scatter(out_steps * DT_MS / 1000, np.ones_like(out_steps), s=8,
               color="tab:blue", label="Output-Spike")
    ax.scatter(pattern_onsets * DT_MS / 1000, np.full_like(pattern_onsets, 1.05),
               s=8, color="orange", marker="|", label="Muster-Onset")
    ax.set_ylim(0.9, 1.15)
    ax.set_xlabel("Zeit [s]")
    ax.set_yticks([])
    ax.set_title("Output-Spikes ueber die gesamte Simulation")
    ax.legend(loc="upper right")

    # (b) Gewichtsentwicklung
    ax = axes[1]
    for ch in range(N_CHANNELS):
        ax.plot(result["weight_log_steps"] / 1000, result["weight_log"][:, ch],
                linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Gewicht")
    ax.set_title("Entwicklung der 16 synaptischen Gewichte")

    # (c) Histogramm: Abstand Output-Spike -> naechster Muster-Onset, frueh vs. spaet
    ax = axes[2]
    bins = np.linspace(0, 150, 76)
    ax.hist(early_offsets, bins=bins, alpha=0.55, label="erste Haelfte der Simulation")
    ax.hist(late_offsets, bins=bins, alpha=0.55, label="zweite Haelfte der Simulation")
    ax.axvline(25, color="black", linestyle="--", linewidth=1,
               label="Landmark-Ereignis (25 ms)")
    ax.set_xlabel("Abstand Output-Spike -> vorheriger Muster-Onset [ms]")
    ax.set_ylabel("Anzahl")
    ax.set_title("Hat sich die Antwortlatenz relativ zum Muster verringert?")
    ax.legend()

    fig.tight_layout()
    fig.savefig("./learning_result.png", dpi=130)
    plt.close(fig)

    print("Finale Gewichte:", np.round(result["final_weights"], 3))
    print("Plots gespeichert: input_raster_pattern_example.png, learning_result.png")


if __name__ == "__main__":
    main()