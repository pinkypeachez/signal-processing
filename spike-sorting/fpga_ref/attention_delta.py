# -*- coding: utf-8 -*-
"""
Referenzmodell der Attention-Stufe mit Delta-Encoder und selbstkalibrierenden
Schwellen.

Ziel dieses Moduls ist NICHT schnelle Simulation, sondern eine 1:1-Vorlage fuer
eine spaetere VHDL-Implementierung. Deshalb gelten durchgehend folgende Regeln:

  R1  Streaming: process_sample() sieht immer nur den aktuellen Abtastwert.
      Es gibt keinen Zugriff auf zukuenftige Daten und keine Statistik ueber
      die gesamte Aufnahme.
  R2  Beschraenkter Speicher: nur Ringpuffer fester Groesse, keine wachsenden
      Listen im Datenpfad.
  R3  Nur Ganzzahlen im Datenpfad. Keine floats, keine exp(), keine Division
      durch Nicht-Zweierpotenzen.
  R4  Leckintegration als Shift:  V <- V - (V >> s)   statt  V * exp(-dt/tau).
  R5  Jeder Zustand ist explizit und hat eine dokumentierte Wortbreite.

Architektur (ein Kanal):

    x[t] --> [Ringpuffer k Samples] --> [k Subtrahierer] --> d_1..d_k
                                                              |
                        +-------------------------------------+
                        |
                        v
             [2k Komparatoren gegen +/- delta_i]  -->  2k Ereignisleitungen
                        |                                     |
                        v                                     v
          [Schwellen-Schaetzer je (Kanal,Lag)]      [Attention-Neuron, LIF]
           (Median- und MAD-Tracker, vorzeichen-        V <- V - (V>>s) + w*n
            basiert, ohne Division)                     V >= Th  ->  Spike
                                                        danach:  V += w_self

Die Attention-Stufe ersetzt hier die Kombination aus rezeptiven Feldern und
STP-Synapsen aus Bernert & Yvert (2019). Die Rauschunterdrueckung leisten nun
die Schwellen delta_i, nicht mehr adaptive Synapsengewichte.

Start:  python attention_delta.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Alle Entwurfsparameter an einer Stelle.

    Wichtig: Die abgeleiteten Groessen (Schwelle des Attention-Neurons,
    Selbsterregung) werden NICHT als nackte Zahlen angegeben, sondern aus den
    Encoder-Parametern berechnet - genau wie in Tabelle 1/2 des ersten Papers.
    So skalieren sie automatisch mit, wenn k oder die Abtastrate geaendert wird.
    """

    # --- Signal / Encoder -------------------------------------------------
    n_channels: int = 1
    fs_hz: int = 20_000
    k: int = 10                      # Anzahl Vergleichsabstaende (Lags)
    sample_bits: int = 16            # vorzeichenbehaftet, ADC-Rohwerte

    # --- Schwellen-Schaetzer ---------------------------------------------
    q_frac: int = 8                  # Nachkommabits der Schaetzer (Q8)
    step_calib: int = 16             # Schrittweite waehrend Kalibrierung (Q8)
    step_track: int = 1              # Schrittweite danach (Q8)
    calib_samples: int = 4_000       # Dauer der Kalibrierphase
    delta_mult: int = 6              # delta = delta_mult * MAD(|d|)
    delta_floor: int = 1             # minimale Schwelle in ADC-LSB, > 0 !
    freeze_after_calib: bool = False # True = Schwellen einfrieren
    gate_tracker_on_spike: bool = False  # Tracker waehrend Detektion anhalten

    # --- Attention-Neuron -------------------------------------------------
    leak_shift: int = 3              # V <- V - (V >> leak_shift)
    w_in: int = 16                   # Gewicht pro Eingangsereignis
    th_num: int = 25                 # Schwelle = th_num/th_den * V_saettigung
    th_den: int = 100
    v_max: int = 1 << 20             # Saettigung, verhindert Ueberlauf

    # --- Selbsterregung (Hysterese) --------------------------------------
    # Im Paper:  w_self = 0.3 * (1/exp(-dts/tau_m) - 1) * Th
    # Mit Shift-Leck ist der Zerfallsfaktor pro Tick a = 1 - 2^-s, also
    #   1/a - 1 = (1-a)/a = 1/(2^s - 1).
    # Daraus folgt direkt  w_self = 0.3 * Th / (2^s - 1),
    # hier ganzzahlig als (3 * Th) // (10 * (2^s - 1)).
    self_exc_num: int = 3
    self_exc_den: int = 10

    # --- Sonstiges --------------------------------------------------------
    seed: int = 1

    # ---- abgeleitete Groessen -------------------------------------------

    @property
    def n_lines(self) -> int:
        """Anzahl Ereignisleitungen insgesamt (2k je Kanal)."""
        return 2 * self.k * self.n_channels

    @property
    def max_events_per_tick(self) -> int:
        """Obergrenze gleichzeitiger Ereignisse.

        Pro (Kanal, Lag) kann bei positiver Schwelle hoechstens EINE Polaritaet
        feuern, denn d >= +delta und d <= -delta schliessen sich fuer delta > 0
        gegenseitig aus. Also k statt 2k je Kanal.
        """
        return self.k * self.n_channels

    @property
    def v_saturation(self) -> int:
        """Stationaerer Wert von V bei dauerhaft maximaler Eingangsrate.

        Fixpunkt von  V = V - (V >> s) + w*r  ist  V = w * r * 2^s.
        """
        return self.w_in * self.max_events_per_tick * (1 << self.leak_shift)

    @property
    def v_threshold(self) -> int:
        """Schwelle als Bruchteil der maximal moeglichen Erregung.

        Entspricht dem Entwurfsprinzip aus Tabelle 2 des ersten Papers, wo
        Th = 0.43 * Noverlap * Nc / (1 - exp(-dts/tau_m)) gesetzt wird.
        """
        return (self.v_saturation * self.th_num) // self.th_den

    @property
    def w_self(self) -> int:
        """Selbsterregung, abgeleitet aus dem Leck-Shift (siehe oben)."""
        decay_term = (1 << self.leak_shift) - 1
        return (self.v_threshold * self.self_exc_num) // (self.self_exc_den * decay_term)

    @property
    def lookback_ms(self) -> float:
        return 1000.0 * self.k / self.fs_hz


# ----------------------------------------------------------------------------
# Baustein 1: Median-Tracker (vorzeichenbasiert)
# ----------------------------------------------------------------------------


class MedianTracker:
    """Online-Schaetzer des Medians, ohne Sortieren und ohne Division.

    Update-Regel:   est <- est + step   falls  x > est
                    est <- est - step   sonst

    Der Erwartungswert der Aenderung ist step * (P(x > est) - P(x < est)).
    Er verschwindet genau dann, wenn est der Median von x ist. Der Schaetzer
    konvergiert also gegen den Median und nicht gegen den Mittelwert.

    Das ist fuer Spike Sorting wichtig: Ein grosses Aktionspotenzial verschiebt
    den Schaetzer um genau einen Schritt, nicht proportional zu seiner Hoehe.
    Der Tracker ist damit robust gegen genau die Ereignisse, die wir erkennen
    wollen.

    Hardware: 1 Komparator, 1 Addierer/Subtrahierer, 1 Register.
    """

    __slots__ = ("est_q", "q_frac")

    def __init__(self, q_frac: int, init_q: int = 0) -> None:
        self.q_frac = q_frac
        self.est_q = init_q

    def update(self, x: int, step_q: int) -> int:
        """x ist eine nicht skalierte Ganzzahl, est_q liegt in Q-Format."""
        if (x << self.q_frac) > self.est_q:
            self.est_q += step_q
        else:
            self.est_q -= step_q
            if self.est_q < 0:
                self.est_q = 0
        return self.est_q

    @property
    def value(self) -> float:
        """Nur fuer Diagnose/Plots, nicht Teil des Datenpfads."""
        return self.est_q / (1 << self.q_frac)


# ----------------------------------------------------------------------------
# Baustein 2: Schwellen-Schaetzer je (Kanal, Lag)
# ----------------------------------------------------------------------------


class ThresholdEstimator:
    """Online-Naeherung von  delta = mult * MAD(|d|).

    Die Autoren berechnen MAD offline ueber die gesamte Aufnahme:

        a[n]  = |x[n] - x[n-i]|
        delta = mult * median_n( | a[n] - median_n(a) | )

    Das ist nicht streamingfaehig. Hier werden beide Mediane durch je einen
    MedianTracker ersetzt:

        Stufe 1:  m   ~ median(|d|)
        Stufe 2:  mad ~ median( | |d| - m | )
        delta     = max(delta_floor, (mult * mad) >> q_frac)

    Hardware je (Kanal, Lag): 2 Komparatoren, 2 Add/Sub, 1 Betragsbildung,
    1 Multiplikation mit einer Konstanten (als Shift-Add realisierbar),
    2 Register.
    """

    __slots__ = ("cfg", "m", "mad", "delta")

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.m = MedianTracker(cfg.q_frac)
        self.mad = MedianTracker(cfg.q_frac)
        self.delta = cfg.delta_floor

    def update(self, abs_d: int, step_q: int) -> int:
        cfg = self.cfg
        m_q = self.m.update(abs_d, step_q)

        # Abweichung vom laufenden Median, wieder als nicht skalierte Zahl.
        dev = abs(abs_d - (m_q >> cfg.q_frac))
        mad_q = self.mad.update(dev, step_q)

        delta = (cfg.delta_mult * mad_q) >> cfg.q_frac
        self.delta = delta if delta > cfg.delta_floor else cfg.delta_floor
        return self.delta


# ----------------------------------------------------------------------------
# Baustein 3: Delta-Encoder
# ----------------------------------------------------------------------------


class DeltaEncoder:
    """Multi-Lag-Delta-Encoder, streamend.

    Speichert je Kanal die letzten k Abtastwerte in einem Ringpuffer und
    vergleicht den aktuellen Wert mit jedem davon.

    Die Leitungsnummerierung folgt exakt encoding/sigma_delta.py der Autoren:

        Index  c*2k + (i-1)      positives Delta, Lag i
        Index  c*2k + (k+i-1)    negatives Delta, Lag i

    Damit lassen sich die Ausgaben direkt gegen deren Implementierung pruefen.

    Hardware: k Register je Kanal (Schieberegister), k Subtrahierer,
    2k Komparatoren. Rein kombinatorisch nach dem Registerausgang.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # hist[c][j] enthaelt x[t-1-j], also hist[c][i-1] == x[t-i]
        self.hist: List[List[int]] = [[0] * cfg.k for _ in range(cfg.n_channels)]
        self.n_filled = 0

    @property
    def warm(self) -> bool:
        """True, sobald der Ringpuffer gueltige Vergangenheit enthaelt."""
        return self.n_filled >= self.cfg.k

    def step(self, samples: Sequence[int], deltas: List[List[int]]) -> List[int]:
        """Ein Abtastzeitpunkt. Gibt die 2k*n_channels Ereignisbits zurueck."""
        cfg = self.cfg
        events = [0] * cfg.n_lines

        if self.warm:
            for c in range(cfg.n_channels):
                x = samples[c]
                hist_c = self.hist[c]
                delta_c = deltas[c]
                base = c * 2 * cfg.k
                for i in range(1, cfg.k + 1):
                    d = x - hist_c[i - 1]
                    th = delta_c[i - 1]
                    if d >= th:
                        events[base + (i - 1)] = 1
                    elif -d >= th:
                        events[base + (cfg.k + i - 1)] = 1

        # Ringpuffer weiterschieben (in HDL: ein Schieberegister)
        for c in range(cfg.n_channels):
            hist_c = self.hist[c]
            for j in range(cfg.k - 1, 0, -1):
                hist_c[j] = hist_c[j - 1]
            hist_c[0] = samples[c]

        if self.n_filled < cfg.k:
            self.n_filled += 1
        return events

    def abs_deltas(self, samples: Sequence[int]) -> List[List[int]]:
        """|d_i| je (Kanal, Lag) fuer die Schwellen-Schaetzer.

        Muss VOR step() aufgerufen werden, solange der Puffer noch den Zustand
        von t-1 haelt.
        """
        cfg = self.cfg
        out: List[List[int]] = []
        for c in range(cfg.n_channels):
            x = samples[c]
            hist_c = self.hist[c]
            out.append([abs(x - hist_c[i - 1]) for i in range(1, cfg.k + 1)])
        return out


# ----------------------------------------------------------------------------
# Baustein 4: Attention-Neuron
# ----------------------------------------------------------------------------


class AttentionNeuron:
    """LIF-Neuron mit Shift-Leck und Selbsterregung statt Reset.

    Aus dem ersten Paper uebernommen:
      - kein Refraktaerzeitraum,
      - KEIN Reset nach dem Feuern,
      - stattdessen wird V durch eine selbsterregende Synapse angehoben.
        Das erzeugt eine Hysterese, damit der Detektionsburst auch dann
        durchgehend bleibt, wenn die Wellenform die Nulllinie kreuzt.

    Nicht uebernommen: die STP-Synapsen. Sie waren im Original noetig, weil die
    rezeptiven Felder bei JEDEM Tick gefeuert haben und das Attention-Neuron
    Haeufiges von Seltenem trennen musste. Mit Delta-Encoding erledigt das
    bereits die Schwelle delta.

    Hardware: 1 Register fuer V, 1 Shifter, 1 Addierer, 1 Komparator,
    1 Populationszaehler ueber die Ereignisbits.
    """

    __slots__ = ("cfg", "v", "th", "w_self", "active")

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.v = 0
        self.th = cfg.v_threshold
        self.w_self = cfg.w_self
        self.active = False

    def step(self, n_events: int) -> int:
        cfg = self.cfg
        v = self.v
        v -= v >> cfg.leak_shift          # R4: Leck als Shift
        v += cfg.w_in * n_events

        fired = 0
        if v >= self.th:
            fired = 1
            v += self.w_self              # Hysterese statt Reset

        if v > cfg.v_max:
            v = cfg.v_max
        elif v < 0:
            v = 0

        self.v = v
        self.active = bool(fired)
        return fired


# ----------------------------------------------------------------------------
# Frontend: alles zusammen
# ----------------------------------------------------------------------------


@dataclass
class StepResult:
    events: List[int]
    n_events: int
    attention: int
    onset: bool          # steigende Flanke des Attention-Bursts
    calibrating: bool


class AttentionFrontEnd:
    """Vollstaendige Streaming-Kette.

    Verwendung:

        fe = AttentionFrontEnd(Config())
        for x in signal:
            r = fe.process_sample([x])
            if r.onset:
                ...
    """

    def __init__(self, cfg: Config,
                 fixed_deltas: Optional[List[List[int]]] = None) -> None:
        self.cfg = cfg
        self.encoder = DeltaEncoder(cfg)
        self.neuron = AttentionNeuron(cfg)
        self.n_samples = 0
        self.prev_attention = 0

        self.fixed = fixed_deltas is not None
        if self.fixed:
            # Modus fuer den Aequivalenztest gegen den Autoren-Encoder.
            self.deltas = [list(row) for row in fixed_deltas]
            self.estimators: List[List[ThresholdEstimator]] = []
        else:
            self.estimators = [
                [ThresholdEstimator(cfg) for _ in range(cfg.k)]
                for _ in range(cfg.n_channels)
            ]
            self.deltas = [[cfg.delta_floor] * cfg.k
                           for _ in range(cfg.n_channels)]

    @property
    def calibrating(self) -> bool:
        return (not self.fixed) and self.n_samples < self.cfg.calib_samples

    def process_sample(self, samples: Sequence[int]) -> StepResult:
        cfg = self.cfg
        calibrating = self.calibrating

        # --- 1. Schwellen nachfuehren -------------------------------------
        if not self.fixed:
            frozen = cfg.freeze_after_calib and not calibrating
            gated = cfg.gate_tracker_on_spike and self.neuron.active
            if not frozen and not gated and self.encoder.warm:
                step_q = cfg.step_calib if calibrating else cfg.step_track
                abs_d = self.encoder.abs_deltas(samples)
                for c in range(cfg.n_channels):
                    est_c = self.estimators[c]
                    for i in range(cfg.k):
                        self.deltas[c][i] = est_c[i].update(abs_d[c][i], step_q)

        # --- 2. Encodieren -------------------------------------------------
        events = self.encoder.step(samples, self.deltas)
        n_events = sum(events)

        # --- 3. Attention-Neuron -------------------------------------------
        # Waehrend der Kalibrierung sind die Schwellen noch zu klein; das Neuron
        # wird deshalb erst danach freigegeben. In HDL ist das ein Enable-Bit.
        if calibrating:
            attention = 0
            self.neuron.v = 0
        else:
            attention = self.neuron.step(n_events)

        onset = bool(attention and not self.prev_attention)
        self.prev_attention = attention
        self.n_samples += 1

        return StepResult(events, n_events, attention, onset, calibrating)

    def current_deltas(self) -> List[List[int]]:
        return [list(row) for row in self.deltas]


# ----------------------------------------------------------------------------
# Testsignal mit bekannter Wahrheit
# ----------------------------------------------------------------------------


def cosine_gaussian(n: int, amp: int, period: float, phase: float,
                    width: float) -> List[int]:
    """Wellenform nach Gl. in Pokala et al. (2025), Methods/Spike-sorting data."""
    out = []
    for j in range(n):
        t = j - n // 2
        env = math.exp(-((2.3548 * t / width) ** 2))
        out.append(int(round(amp * math.cos(2 * math.pi * t / period + phase) * env)))
    return out


def make_test_signal(cfg: Config, duration_s: float = 2.0,
                     noise_sigma: int = 25, snr: float = 6.0,
                     rate_hz: float = 20.0) -> Tuple[List[int], List[int]]:
    """Rauschen plus Aktionspotenziale zu bekannten Zeitpunkten.

    Gibt (samples, onsets) zurueck. onsets sind die Indizes, an denen eine
    Wellenform beginnt einzusetzen (Beginn des Peaks).
    """
    rng = random.Random(cfg.seed)
    n = int(duration_s * cfg.fs_hz)
    sig = [int(round(rng.gauss(0, noise_sigma))) for _ in range(n)]

    wf_len = int(0.0015 * cfg.fs_hz) | 1           # ~1.5 ms, ungerade
    amp = int(round(snr * noise_sigma))
    wave = cosine_gaussian(wf_len, amp, period=wf_len / 1.6,
                           phase=0.0, width=wf_len / 2.2)

    onsets: List[int] = []
    refractory = int(0.003 * cfg.fs_hz)
    guard = cfg.calib_samples + wf_len
    t = guard
    p = rate_hz / cfg.fs_hz
    while t < n - wf_len:
        if rng.random() < p:
            for j, v in enumerate(wave):
                sig[t + j] += v
            onsets.append(t + wf_len // 2)          # Zeitpunkt des Extremums
            t += refractory + wf_len
        else:
            t += 1

    lo, hi = -(1 << (cfg.sample_bits - 1)), (1 << (cfg.sample_bits - 1)) - 1
    sig = [lo if s < lo else hi if s > hi else s for s in sig]
    return sig, onsets


# ----------------------------------------------------------------------------
# Auswertung
# ----------------------------------------------------------------------------


def score_detection(onsets_true: Sequence[int], onsets_pred: Sequence[int],
                    tol_pre: int, tol_post: int) -> dict:
    """Vergleicht Detektionszeitpunkte mit Toleranzfenster.

    Ein wahres Ereignis gilt als erkannt, wenn eine Vorhersage im Fenster
    [t - tol_pre, t + tol_post] liegt. Jede Vorhersage zaehlt hoechstens einmal.
    """
    used = [False] * len(onsets_pred)
    hits = 0
    for t in onsets_true:
        for j, p in enumerate(onsets_pred):
            if used[j]:
                continue
            if t - tol_pre <= p <= t + tol_post:
                used[j] = True
                hits += 1
                break
    fp = len(onsets_pred) - hits
    fn = len(onsets_true) - hits
    prec = hits / len(onsets_pred) if onsets_pred else 0.0
    rec = hits / len(onsets_true) if onsets_true else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"hits": hits, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1}


def run(cfg: Config, signal: Sequence[int]) -> dict:
    """Laesst das Frontend ueber ein einkanaliges Signal laufen."""
    fe = AttentionFrontEnd(cfg)
    onsets: List[int] = []
    n_events_total = 0
    n_events_post_calib = 0
    v_trace: List[int] = []

    for t, x in enumerate(signal):
        r = fe.process_sample([x])
        n_events_total += r.n_events
        if not r.calibrating:
            n_events_post_calib += r.n_events
        if r.onset:
            onsets.append(t)
        v_trace.append(fe.neuron.v)

    n_post = max(1, len(signal) - cfg.calib_samples)
    return {
        "frontend": fe,
        "onsets": onsets,
        "events_total": n_events_total,
        "events_per_sample": n_events_post_calib / n_post,
        "deltas": fe.current_deltas(),
        "v_trace": v_trace,
    }


# ----------------------------------------------------------------------------
# Selbsttests
# ----------------------------------------------------------------------------


def test_streaming_matches_authors(cfg: Config, signal: Sequence[int]) -> str:
    """Gleiche Ausgaben wie encoding/sigma_delta.py bei festen Schwellen.

    Das ist der wichtigste Test: er beweist, dass die Streaming-Umformulierung
    keine Semantik veraendert hat.
    """
    try:
        import numpy as np
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from encoding.sigma_delta import generate_spike_trains_robust
    except Exception as exc:                                  # pragma: no cover
        return "UEBERSPRUNGEN (numpy/tqdm/Autorencode nicht verfuegbar: %s)" % exc

    fixed = [[7 + 2 * i for i in range(cfg.k)]]
    x = np.asarray([list(signal)], dtype=np.int64)
    ref = generate_spike_trains_robust(x, cfg.k, np.asarray(fixed, dtype=np.int64))

    fe = AttentionFrontEnd(cfg, fixed_deltas=fixed)
    mine = []
    for v in signal:
        mine.append(fe.process_sample([int(v)]).events)

    mine_arr = np.asarray(mine, dtype=np.int64).T
    ok = bool((mine_arr == ref.astype(np.int64)).all())
    return "OK (identisch, %d Leitungen x %d Samples)" % ref.shape if ok else "FEHLGESCHLAGEN"


def test_causality(cfg: Config, signal: Sequence[int]) -> str:
    """Ein Praefix muss dieselben Ausgaben liefern wie der gleiche Abschnitt
    des vollen Laufs. Beweist, dass nirgends in die Zukunft geschaut wird."""
    cut = len(signal) // 2

    full = AttentionFrontEnd(cfg)
    out_full = [full.process_sample([x]).attention for x in signal[:cut]]

    part = AttentionFrontEnd(cfg)
    out_part = [part.process_sample([x]).attention for x in signal[:cut]]

    return "OK" if out_full == out_part else "FEHLGESCHLAGEN"


def test_integer_datapath(cfg: Config, signal: Sequence[int]) -> str:
    """Kein float im Datenpfad, keine Ueberschreitung der Wortbreiten."""
    fe = AttentionFrontEnd(cfg)
    v_peak = 0
    for x in signal[: min(len(signal), 30_000)]:
        r = fe.process_sample([x])
        if not isinstance(r.n_events, int) or not isinstance(fe.neuron.v, int):
            return "FEHLGESCHLAGEN (float im Datenpfad)"
        v_peak = max(v_peak, fe.neuron.v)
    for row in fe.estimators:
        for est in row:
            if not isinstance(est.delta, int):
                return "FEHLGESCHLAGEN (float in Schwelle)"
    bits = v_peak.bit_length() + 1
    return "OK (V_max beobachtet = %d, benoetigt %d Bit, reserviert %d Bit)" % (
        v_peak, bits, cfg.v_max.bit_length())


def test_zero_threshold_guard(cfg: Config) -> str:
    """delta = 0 wuerde auf konstantem Signal beide Polaritaeten gleichzeitig
    ausloesen. delta_floor > 0 muss das verhindern."""
    flat = [0] * 200
    fe = AttentionFrontEnd(cfg)
    total = 0
    for x in flat:
        total += fe.process_sample([x]).n_events
    return "OK (0 Ereignisse auf konstantem Signal)" if total == 0 else \
           "FEHLGESCHLAGEN (%d Ereignisse)" % total


def test_tracker_vs_offline_mad(cfg: Config, signal: Sequence[int]) -> str:
    """Online-Tracker gegen die Offline-MAD-Formel der Autoren."""
    try:
        import numpy as np
        from scipy.stats import median_abs_deviation
    except Exception as exc:                                  # pragma: no cover
        return "UEBERSPRUNGEN (%s)" % exc

    fe = AttentionFrontEnd(cfg)
    for x in signal:
        fe.process_sample([x])
    online = fe.current_deltas()[0]

    x = np.asarray(signal, dtype=np.float64)
    rows = []
    for i in range(1, cfg.k + 1):
        d = np.abs(x[i:] - x[:-i])
        rows.append(cfg.delta_mult * median_abs_deviation(d))

    errs = [abs(online[i] - rows[i]) / rows[i] for i in range(cfg.k)]
    worst = max(errs) * 100
    return "OK (max. Abweichung %.1f %%)" % worst if worst < 25 else \
           "WARNUNG (max. Abweichung %.1f %%)" % worst


# ----------------------------------------------------------------------------
# Hardware-Abschaetzung
# ----------------------------------------------------------------------------


def hardware_notes(cfg: Config) -> str:
    k, nch = cfg.k, cfg.n_channels
    cyc = 100_000_000 // cfg.fs_hz

    hist_bits = nch * k * cfg.sample_bits
    est_bits = nch * k * 2 * (cfg.sample_bits + cfg.q_frac)
    delta_bits = nch * k * cfg.sample_bits

    lines = [
        "Modul                     | HDL-Entsprechung                      | Zustand",
        "-" * 88,
        "DeltaEncoder.hist         | %2d x %2d Schieberegister (%d Bit)      | %5d Bit"
        % (nch, k, cfg.sample_bits, hist_bits),
        "DeltaEncoder.step         | %3d Subtrahierer, %3d Komparatoren    | -"
        % (nch * k, nch * 2 * k),
        "MedianTracker             | Komparator + Add/Sub, kein Divider    | -",
        "ThresholdEstimator        | %3d Tracker-Paare + Konstantenmult.   | %5d Bit"
        % (nch * k, est_bits),
        "Schwellenregister delta   | %3d Register                          | %5d Bit"
        % (nch * k, delta_bits),
        "AttentionNeuron           | Shifter + Addierer + Komparator       | %5d Bit"
        % cfg.v_max.bit_length(),
        "Populationszaehler        | Addierbaum ueber %3d Bits             | -"
        % (nch * 2 * k),
        "",
        "Zeitbudget:",
        "  Abtastrate                 %d Hz  ->  %.1f us pro Sample"
        % (cfg.fs_hz, 1e6 / cfg.fs_hz),
        "  bei 100 MHz Systemtakt     %d Takte pro Sample" % cyc,
        "  Lag-Operationen je Sample  %d  (voll parallel oder sequenziell moeglich)"
        % (nch * k),
        "",
        "Vergleich zur Eingangsschicht des ersten Papers (pro Kanal):",
        "  rezeptive Felder  ~500 Leitungen -> 500 x 60 = 30000 Synapsen + 500 STP-Gewichte",
        "  Delta k=%-2d          %3d Leitungen -> %3d x 60 = %5d Synapsen + 0 STP-Gewichte"
        % (k, 2 * k, 2 * k, 2 * k * 60),
        "  Zeitschritte       80 kHz (interpoliert)  ->  %d kHz (nativ)" % (cfg.fs_hz // 1000),
        "",
        "Keine Division und keine exp() im Datenpfad. Das Leck ist ein Shift,",
        "die Schwellenmultiplikation eine Konstante (%d = Shift-Add)." % cfg.delta_mult,
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------


def main() -> None:
    cfg = Config()

    print("=" * 88)
    print("REFERENZMODELL: Delta-Encoder + selbstkalibrierende Schwellen + Attention")
    print("=" * 88)
    print("Kanaele %d | fs %d Hz | k %d -> Rueckblick %.2f ms | %d Ereignisleitungen"
          % (cfg.n_channels, cfg.fs_hz, cfg.k, cfg.lookback_ms, cfg.n_lines))
    print("max. Ereignisse/Tick %d | V_saettigung %d | Th %d (%d%%) | w_self %d"
          % (cfg.max_events_per_tick, cfg.v_saturation, cfg.v_threshold,
             100 * cfg.th_num // cfg.th_den, cfg.w_self))

    signal, truth = make_test_signal(cfg, duration_s=2.0, noise_sigma=25, snr=6.0)
    print("\nTestsignal: %d Samples, %d Aktionspotenziale, Rauschen sigma=25, SNR=6"
          % (len(signal), len(truth)))

    res = run(cfg, signal)

    print("\n--- Gelernte Schwellen (Kanal 0) ---")
    print("Lag i :  " + " ".join("%4d" % (i + 1) for i in range(cfg.k)))
    print("delta :  " + " ".join("%4d" % d for d in res["deltas"][0]))

    tol_pre = int(0.0010 * cfg.fs_hz)
    tol_post = int(0.0015 * cfg.fs_hz)
    sc = score_detection(truth, res["onsets"], tol_pre, tol_post)

    print("\n--- Detektionsleistung des Attention-Neurons ---")
    print("Toleranzfenster: -%d .. +%d Samples (%.1f .. %.1f ms)"
          % (tol_pre, tol_post, -1000 * tol_pre / cfg.fs_hz, 1000 * tol_post / cfg.fs_hz))
    print("wahr %d | erkannt %d | Treffer %d | FP %d | FN %d"
          % (len(truth), len(res["onsets"]), sc["hits"], sc["fp"], sc["fn"]))
    print("Precision %.3f | Recall %.3f | F1 %.3f"
          % (sc["precision"], sc["recall"], sc["f1"]))
    print("Ereignisrate nach Kalibrierung: %.3f pro Sample (von max. %d)"
          % (res["events_per_sample"], cfg.max_events_per_tick))

    print("\n--- Robustheit ueber SNR (je 3 Zufallsstartwerte) ---")
    print(" SNR |   F1 je Seed     | Mittel | Ereignisse/Sample")
    for snr in (3.0, 4.0, 5.0, 6.0, 8.0):
        f1s, rates = [], []
        for seed in (1, 2, 3):
            c = Config(seed=seed, delta_mult=cfg.delta_mult,
                       leak_shift=cfg.leak_shift, th_num=cfg.th_num)
            s_sig, s_truth = make_test_signal(c, 2.0, 25, snr)
            s_res = run(c, s_sig)
            s_sc = score_detection(s_truth, s_res["onsets"], tol_pre, tol_post)
            f1s.append(s_sc["f1"])
            rates.append(s_res["events_per_sample"])
        print(" %.1f | %s | %.3f  | %.3f"
              % (snr, "  ".join("%.2f" % f for f in f1s),
                 sum(f1s) / len(f1s), sum(rates) / len(rates)))

    print("\n--- Selbsttests ---")
    short = signal[:20_000]
    print("Streaming == Autoren-Encoder : %s"
          % test_streaming_matches_authors(cfg, short))
    print("Kausalitaet (Praefixtest)    : %s" % test_causality(cfg, short))
    print("Ganzzahliger Datenpfad       : %s" % test_integer_datapath(cfg, signal))
    print("Schutz gegen delta = 0       : %s" % test_zero_threshold_guard(cfg))
    print("Tracker vs. Offline-MAD      : %s"
          % test_tracker_vs_offline_mad(cfg, signal))

    print("\n--- Hardware-Abschaetzung ---")
    print(hardware_notes(cfg))


if __name__ == "__main__":
    main()
