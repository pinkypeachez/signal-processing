# -*- coding: utf-8 -*-
"""
Vollstaendiger Spike-Sorter nach Bernert & Yvert (2019), aber mit dem
Delta-Encoder aus Pokala et al. (2025) als Eingangsstufe.

Baut auf attention_delta.py auf und ergaenzt die beiden Lernschichten:

    x[t] -> DeltaEncoder -> [2k Ereignisbits]
                               |            |
                               |            +-> AttentionNeuron ---+
                               v                                   |
                        ZWISCHENSCHICHT (LIF + STDP + WTA)  <------+ (Gating)
                               |
                               v
                        [Verzoegerungsleitungen, Nd Abgriffe]
                               |
                               |  praesynaptische Inhibition durch Attention
                               v
                        AUSGABESCHICHT (LIF + STDP + IP + WTA)
                               |
                               v
                        1 Spike je Aktionspotenzial, je Zelle ein Neuron


Der zentrale Mechanismus der Ausgabeschicht
-------------------------------------------
Das Attention-Neuron blockiert die ankommenden Zwischenspikes, SOLANGE es
selbst feuert. Da es waehrend des gesamten Aktionspotenzials aktiv ist,
erreichen die verzoegerten Zwischenspikes die Ausgabeschicht erst NACH dem
Ende des Musters. Dadurch wartet die Ausgabeschicht automatisch das komplette
Aktionspotenzial ab, bevor sie sich entscheidet.

Das ist die Funktion, die im Nachfolgepaper von den LTS-Neuronen uebernommen
wird - dort ohne Verzoegerungssynapsen.


Ereignisbasierte Verarbeitung (Antwort auf die AER-Frage)
---------------------------------------------------------
Alle Synapsenschleifen iterieren ueber die INDIZES aktiver Leitungen, nicht
ueber alle Leitungen. In VHDL entspricht das einem Priority Encoder, der das
gefundene Bit anschliessend loescht. Es wird KEIN Adressbus und kein
AER-Handshake benoetigt - die Schnittstelle zwischen den Modulen bleibt ein
dichter Bitvektor.

Die Zaehler op_dense und op_sparse belegen die Ersparnis quantitativ.


Abweichungen vom Original (bewusst und dokumentiert)
----------------------------------------------------
 A1  Das Paper nutzt in der Ausgabeschicht je Verzoegerung EINE exzitatorische
     UND eine inhibitorische Synapse, deren Summe gegen +1, 0 oder -1
     konvergiert. Hier ist stattdessen ein einzelnes vorzeichenbehaftetes
     Gewicht implementiert (zwei Attraktoren statt drei). Grund: Supplementary
     Table 1D mit den exakten Verhaeltnissen liegt nicht vor.
 A2  Schichtgroessen verkleinert (20 statt 60 Zwischen-, 6 statt 10
     Ausgabeneuronen), damit die Referenz in Sekunden statt Minuten laeuft.
 A3  Zeitkonstanten sind auf das Zeitraster des Delta-Encoders umgerechnet
     (50 us statt 12.5 us pro Tick), nicht direkt uebernommen.

Start:  python sorter.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from attention_delta import (
    AttentionNeuron,
    Config,
    DeltaEncoder,
    ThresholdEstimator,
    cosine_gaussian,
)

# Festkommaskala fuer Synapsengewichte: 1.0 entspricht W_ONE.
W_ONE = 256


# ----------------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SorterConfig:
    front: Config = Config()

    # --- Zwischenschicht --------------------------------------------------
    n_inter: int = 20
    im_leak_shift: int = 2           # V <- V - (V >> s)
    im_refractory: int = 3           # Ticks; Analogon zu tau_refrac = dtc
    im_w_att_num: int = 60           # w_attention = num/den * k * W_ONE
    im_w_att_den: int = 100
    im_th_num: int = 22              # Schwelle als Bruchteil der Saettigung
    im_th_den: int = 100
    im_stdp_window: int = 3          # Ticks; Analogon zu tau_stdp+
    # Der Delta-Encoder liefert im Mittel nur ~3 aktive Leitungen pro Tick,
    # der Amplitudenraster des Originals dagegen 100. Der Zwischenschicht
    # fehlt dadurch Information pro Zeitschritt. Mit im_use_recency sieht sie
    # statt der Momentaufnahme das Fenster "welche Leitung war in den letzten
    # W Ticks aktiv". In Hardware kostet das nichts, weil die Abklingzaehler
    # fuer die STDP-Koinzidenz ohnehin existieren.
    im_use_recency: bool = True
    im_input_window: int = 4         # W fuer die obige Anreicherung
    im_dw_pair: int = 3              # LTP-Schritt in Gewichts-LSB
    im_dw_post_num: int = 55         # LTD = -0.55 * dw_pair  (Paper)
    im_dw_post_den: int = 100
    im_w_init_lo: int = 102          # 0.4 * W_ONE   (Paper: winit- = 0.4)
    im_w_init_hi: int = 256          # 1.0 * W_ONE   (Paper: winit+ = 1.0)

    # --- Verzoegerungsleitungen -------------------------------------------
    n_delays: int = 8
    delay_step: int = 4              # Ticks zwischen zwei Abgriffen

    # --- Ausgabeschicht ---------------------------------------------------
    n_output: int = 6
    ol_leak_shift: int = 4
    ol_refractory: int = 20
    ol_stdp_window: int = 6
    ol_dw_pair: int = 8
    ol_dw_post_num: int = 55
    ol_dw_post_den: int = 100
    ol_w_init: int = 24              # kleine positive Startgewichte
    # Das Paper initialisiert die Schwelle AUF dem Maximum. Das funktioniert
    # nur, wenn das Maximum ueberhaupt erreichbar ist. Hier sind Startwert und
    # Obergrenze getrennt: ol_th_init liegt im messbar erreichbaren Bereich
    # (~380 mit den Startgewichten), ol_th_max laesst der IP danach Luft nach
    # oben, wenn die Gewichte durch STDP wachsen.
    ol_th_init: int = 250
    ol_th_min: int = 32
    ol_th_max: int = 1 << 15
    ol_ip_post_shift: int = 3        # Th -= Th >> shift   (multiplikativ)
    ol_ip_pair_shift: int = 5        # Th += w >> shift    je Koinzidenz

    seed: int = 7

    # ---- abgeleitete Groessen -------------------------------------------

    @property
    def n_lines(self) -> int:
        return self.front.n_lines

    @property
    def im_w_attention(self) -> int:
        """Analog zu wAN = 0.6 * Nc * Noverlap im Original."""
        return (self.im_w_att_num * self.front.max_events_per_tick * W_ONE) \
            // self.im_w_att_den

    @property
    def im_drive_max(self) -> int:
        """Maximale Erregung pro Tick: alle Leitungen aktiv plus Attention."""
        return self.front.max_events_per_tick * W_ONE + self.im_w_attention

    @property
    def im_threshold(self) -> int:
        v_sat = self.im_drive_max << self.im_leak_shift
        return (v_sat * self.im_th_num) // self.im_th_den

    @property
    def max_delay(self) -> int:
        return self.n_delays * self.delay_step


# ----------------------------------------------------------------------------
# Hilfsmittel: Abklingzaehler statt Bit-Historie
# ----------------------------------------------------------------------------


class RecencyCounters:
    """'War Leitung i in den letzten W Ticks aktiv?'

    Statt eine Bit-Historie der Tiefe W zu speichern und zu durchsuchen (so
    macht es der Python-Code der Autoren), haelt jede Leitung einen kleinen
    Abwaertszaehler:

        aktiv  -> Zaehler = W
        sonst  -> Zaehler = max(0, Zaehler - 1)

    'aktiv im Fenster' ist dann einfach Zaehler > 0.

    Hardware: ein Zaehler mit log2(W)+1 Bit je Leitung, ein Dekrement pro Tick.
    Das ersetzt das wiederholte Durchsuchen der Vergangenheit vollstaendig.
    """

    __slots__ = ("n", "window", "c")

    def __init__(self, n: int, window: int) -> None:
        self.n = n
        self.window = window
        self.c = [0] * n

    def tick(self, active_idx: Sequence[int]) -> None:
        c = self.c
        for i in range(self.n):
            if c[i]:
                c[i] -= 1
        for i in active_idx:
            c[i] = self.window

    def recent(self) -> List[int]:
        return [i for i, v in enumerate(self.c) if v]


def bits_to_indices(bits: Sequence[int]) -> List[int]:
    """Entspricht dem Priority Encoder in VHDL: liefert die gesetzten Indizes."""
    return [i for i, b in enumerate(bits) if b]


# ----------------------------------------------------------------------------
# Zwischenschicht
# ----------------------------------------------------------------------------


class IntermediateLayer:
    """LIF-Neuronen mit STDP auf den Encoder-Leitungen und globalem WTA.

    Aufgabe laut Paper: Fragmente der Wellenform lernen, sodass fuer jede
    Aktionspotenzialform eine charakteristische Spikefolge entsteht.

    Uebernommen:
      - Gating durch das Attention-Neuron (feste Synapse w_attention)
      - WTA: feuert ein Neuron, werden ALLE Potenziale auf 0 gesetzt
      - Refraktaerzeit, damit die Neuronen nacheinander feuern
      - STDP: jeder postsynaptische Spike senkt alle Gewichte um dw_post;
        koinzidente Eingaenge erhalten zusaetzlich +dw_pair
    """

    def __init__(self, cfg: SorterConfig) -> None:
        self.cfg = cfg
        rng = random.Random(cfg.seed)
        n_in = cfg.n_lines
        self.w = [[rng.randint(cfg.im_w_init_lo, cfg.im_w_init_hi)
                   for _ in range(n_in)] for _ in range(cfg.n_inter)]
        self.v = [0] * cfg.n_inter
        self.refrac = [0] * cfg.n_inter
        self.th = cfg.im_threshold
        self.rec = RecencyCounters(n_in, cfg.im_stdp_window)
        self.inp = RecencyCounters(n_in, cfg.im_input_window)
        self.dw_post = -(cfg.im_dw_pair * cfg.im_dw_post_num) // cfg.im_dw_post_den
        self.op_dense = 0
        self.op_sparse = 0

    def step(self, active_in: Sequence[int], attention: int,
             learn: bool = True) -> int:
        """Gibt den Index des feuernden Neurons zurueck, sonst -1."""
        cfg = self.cfg
        n_in = cfg.n_lines
        self.rec.tick(active_in)
        self.inp.tick(active_in)
        if cfg.im_use_recency:
            active_in = self.inp.recent()

        drive = 0
        # Ereignisbasiert: nur aktive Leitungen anfassen.
        self.op_dense += cfg.n_inter * n_in
        self.op_sparse += cfg.n_inter * len(active_in)

        best, best_v = -1, -1
        for j in range(cfg.n_inter):
            if self.refrac[j]:
                self.refrac[j] -= 1
                self.v[j] = 0
                continue
            v = self.v[j]
            v -= v >> cfg.im_leak_shift
            wj = self.w[j]
            drive = 0
            for i in active_in:
                drive += wj[i]
            if attention:
                drive += cfg.im_w_attention
            v += drive
            if v < 0:
                v = 0
            self.v[j] = v
            if v >= self.th and v > best_v:
                best, best_v = j, v

        if best < 0:
            return -1

        # --- WTA: alle zuruecksetzen ---------------------------------------
        for j in range(cfg.n_inter):
            self.v[j] = 0
        self.refrac[best] = cfg.im_refractory

        # --- STDP nur auf dem Gewinner --------------------------------------
        if learn:
            wj = self.w[best]
            dwp = self.dw_post
            for i in range(n_in):
                nw = wj[i] + dwp
                wj[i] = 0 if nw < 0 else nw
            for i in self.rec.recent():
                nw = wj[i] + cfg.im_dw_pair
                wj[i] = W_ONE if nw > W_ONE else nw

        return best


# ----------------------------------------------------------------------------
# Verzoegerungsleitungen mit praesynaptischer Inhibition
# ----------------------------------------------------------------------------


class DelayLines:
    """Ringpuffer der Zwischenspikes mit Nd Abgriffen.

    Liefert je Tick die Liste der Paare (Zwischenneuron, Verzoegerungsindex),
    deren Spike JETZT ankommt - aber nur, wenn das Attention-Neuron gerade
    NICHT feuert (praesynaptische Inhibition nach der Verzoegerung, genau wie
    im Paper beschrieben).

    Hardware: ein Ringpuffer der Tiefe n_delays*delay_step mit n_inter Bit
    Breite. Die Abgriffe sind feste Offsets, also nur Adressrechnung.
    """

    def __init__(self, cfg: SorterConfig) -> None:
        self.cfg = cfg
        self.depth = cfg.max_delay + 1
        self.buf: List[List[int]] = [[] for _ in range(self.depth)]
        self.pos = 0

    def step(self, fired: int, attention: int) -> List[Tuple[int, int]]:
        cfg = self.cfg
        self.buf[self.pos] = [fired] if fired >= 0 else []

        arrivals: List[Tuple[int, int]] = []
        if not attention:                       # praesynaptische Inhibition
            for d in range(cfg.n_delays):
                lag = (d + 1) * cfg.delay_step
                idx = (self.pos - lag) % self.depth
                for j in self.buf[idx]:
                    arrivals.append((j, d))

        self.pos = (self.pos + 1) % self.depth
        return arrivals


# ----------------------------------------------------------------------------
# Ausgabeschicht
# ----------------------------------------------------------------------------


class OutputLayer:
    """LIF mit vorzeichenbehafteten Gewichten, STDP, Intrinsic Plasticity, WTA.

    Die Intrinsic Plasticity ist der Schluessel fuer Muster unterschiedlicher
    Laenge (Paper, Abschnitt Output layer):

        bei jedem eigenen Spike:      Th <- Th - F * Th
        je koinzidentem Eingang:      Th <- Th + dTh_pair * w

    Der Gleichgewichtswert ist damit proportional zur mittleren gewichteten
    Zahl der Eingangsspikes im Koinzidenzfenster - also zur Mustergroesse.
    Ein Neuron, das ein langes Muster gelernt hat, bekommt eine hohe Schwelle
    und feuert nicht mehr auf unvollstaendige Muster.

    Hardware: Th -= Th >> shift ist ein Shift und ein Subtrahierer,
    Th += w >> shift ebenso. Keine Division.
    """

    def __init__(self, cfg: SorterConfig) -> None:
        self.cfg = cfg
        rng = random.Random(cfg.seed + 1)
        self.n_syn = cfg.n_inter * cfg.n_delays
        self.w = [[rng.randint(0, cfg.ol_w_init) for _ in range(self.n_syn)]
                  for _ in range(cfg.n_output)]
        self.v = [0] * cfg.n_output
        self.th = [cfg.ol_th_init] * cfg.n_output
        self.refrac = [0] * cfg.n_output
        self.rec = RecencyCounters(self.n_syn, cfg.ol_stdp_window)
        self.dw_post = -(cfg.ol_dw_pair * cfg.ol_dw_post_num) // cfg.ol_dw_post_den
        self.op_dense = 0
        self.op_sparse = 0

    def step(self, arrivals: Sequence[Tuple[int, int]],
             learn: bool = True) -> int:
        cfg = self.cfg
        syn_idx = [j * cfg.n_delays + d for (j, d) in arrivals]
        self.rec.tick(syn_idx)

        self.op_dense += cfg.n_output * self.n_syn
        self.op_sparse += cfg.n_output * len(syn_idx)

        best, best_margin = -1, 0
        for o in range(cfg.n_output):
            if self.refrac[o]:
                self.refrac[o] -= 1
                self.v[o] = 0
                continue
            v = self.v[o]
            v -= v >> cfg.ol_leak_shift
            wo = self.w[o]
            for s in syn_idx:
                v += wo[s]
            if v < 0:
                v = 0
            self.v[o] = v
            margin = v - self.th[o]
            if margin >= 0 and (best < 0 or margin > best_margin):
                best, best_margin = o, margin

        if best < 0:
            return -1

        for o in range(cfg.n_output):
            self.v[o] = 0
        self.refrac[best] = cfg.ol_refractory

        if learn:
            wo = self.w[best]
            dwp = self.dw_post
            for s in range(self.n_syn):
                nw = wo[s] + dwp
                wo[s] = -W_ONE if nw < -W_ONE else nw

            th = self.th[best]
            th -= th >> cfg.ol_ip_post_shift          # multiplikative Senkung
            for s in self.rec.recent():
                nw = wo[s] + cfg.ol_dw_pair
                wo[s] = W_ONE if nw > W_ONE else nw
                th += wo[s] >> cfg.ol_ip_pair_shift   # Anhebung je Koinzidenz
            if th < cfg.ol_th_min:
                th = cfg.ol_th_min
            elif th > cfg.ol_th_max:
                th = cfg.ol_th_max
            self.th[best] = th

        return best


# ----------------------------------------------------------------------------
# Gesamtnetz
# ----------------------------------------------------------------------------


@dataclass
class SortStep:
    events: List[int]
    attention: int
    inter: int
    output: int
    calibrating: bool


class SpikeSorter:
    """Encoder + Attention + Zwischenschicht + Verzoegerungen + Ausgabeschicht."""

    def __init__(self, cfg: SorterConfig) -> None:
        self.cfg = cfg
        f = cfg.front
        self.encoder = DeltaEncoder(f)
        self.neuron = AttentionNeuron(f)
        self.estimators = [[ThresholdEstimator(f) for _ in range(f.k)]
                           for _ in range(f.n_channels)]
        self.deltas = [[f.delta_floor] * f.k for _ in range(f.n_channels)]
        self.inter = IntermediateLayer(cfg)
        self.delays = DelayLines(cfg)
        self.out = OutputLayer(cfg)
        self.n_samples = 0

    @property
    def calibrating(self) -> bool:
        return self.n_samples < self.cfg.front.calib_samples

    def process_sample(self, samples: Sequence[int]) -> SortStep:
        f = self.cfg.front
        calibrating = self.calibrating

        # 1. Schwellen nachfuehren
        if self.encoder.warm:
            step_q = f.step_calib if calibrating else f.step_track
            abs_d = self.encoder.abs_deltas(samples)
            for c in range(f.n_channels):
                for i in range(f.k):
                    self.deltas[c][i] = self.estimators[c][i].update(
                        abs_d[c][i], step_q)

        # 2. Encodieren
        events = self.encoder.step(samples, self.deltas)
        active = bits_to_indices(events)          # <- Priority Encoder

        self.n_samples += 1
        if calibrating:
            self.neuron.v = 0
            return SortStep(events, 0, -1, -1, True)

        # 3. Attention
        attention = self.neuron.step(len(active))

        # 4. Zwischenschicht (nur waehrend Attention aktiv, wie im Paper)
        fired = self.inter.step(active, attention) if attention else -1

        # 5. Verzoegerung + praesynaptische Inhibition
        arrivals = self.delays.step(fired, attention)

        # 6. Ausgabeschicht
        out = self.out.step(arrivals)

        return SortStep(events, attention, fired, out, False)


# ----------------------------------------------------------------------------
# Mehrzelliges Testsignal
# ----------------------------------------------------------------------------


def make_multiunit_signal(cfg: SorterConfig, duration_s: float = 6.0,
                          n_units: int = 3, noise_sigma: int = 25,
                          snr: float = 7.0, rate_hz: float = 12.0
                          ) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Rauschen plus n_units unterscheidbare Wellenformen.

    Rueckgabe: (samples, [(zeitpunkt, unit_id), ...])
    """
    f = cfg.front
    rng = random.Random(cfg.seed + 99)
    n = int(duration_s * f.fs_hz)
    sig = [int(round(rng.gauss(0, noise_sigma))) for _ in range(n)]

    wf_len = int(0.0015 * f.fs_hz) | 1
    base_amp = snr * noise_sigma
    waves = []
    # Die Zellen unterscheiden sich vor allem in der FORM (Periode, Phase,
    # Huellkurvenbreite), nur schwach in der Amplitude. Sonst testet man
    # ueberwiegend die SNR-Grenze des Detektors statt der Sortierleistung.
    for u in range(n_units):
        amp = int(round(base_amp * (1.0 - 0.08 * u)))
        period = wf_len / (1.25 + 0.45 * u)
        phase = 0.7 * u
        width = wf_len / (1.9 + 0.55 * u)
        waves.append(cosine_gaussian(wf_len, amp, period, phase, width))

    truth: List[Tuple[int, int]] = []
    refractory = int(0.004 * f.fs_hz)
    t = f.calib_samples + wf_len
    p = rate_hz * n_units / f.fs_hz
    while t < n - wf_len:
        if rng.random() < p:
            u = rng.randrange(n_units)
            for j, v in enumerate(waves[u]):
                sig[t + j] += v
            truth.append((t + wf_len // 2, u))
            t += refractory + wf_len
        else:
            t += 1

    lo, hi = -(1 << (f.sample_bits - 1)), (1 << (f.sample_bits - 1)) - 1
    sig = [lo if s < lo else hi if s > hi else s for s in sig]
    return sig, truth


# ----------------------------------------------------------------------------
# Auswertung
# ----------------------------------------------------------------------------


def match_and_score(truth: Sequence[Tuple[int, int]],
                    preds: Sequence[Tuple[int, int]],
                    n_units: int, n_output: int,
                    tol_pre: int, tol_post: int) -> dict:
    """Ordnet Ausgabeneuronen den wahren Zellen zu und bewertet.

    Vorgehen wie im Nachfolgepaper: Trefferzahlen fuer jedes Paar bestimmen,
    dann die Zuordnung mit maximaler Trefferzahl waehlen (hier gierig, falls
    scipy fehlt).
    """
    hits = [[0] * n_output for _ in range(n_units)]
    for u in range(n_units):
        t_u = [t for (t, uu) in truth if uu == u]
        for o in range(n_output):
            p_o = [t for (t, oo) in preds if oo == o]
            used = [False] * len(p_o)
            c = 0
            for t in t_u:
                for j, p in enumerate(p_o):
                    if not used[j] and t - tol_pre <= p <= t + tol_post:
                        used[j] = True
                        c += 1
                        break
            hits[u][o] = c

    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-np.asarray(hits))
        pairs = dict(zip(rows.tolist(), cols.tolist()))
    except Exception:
        pairs, taken = {}, set()
        order = sorted(range(n_units),
                       key=lambda u: -max(hits[u]) if hits[u] else 0)
        for u in order:
            best, bv = -1, -1
            for o in range(n_output):
                if o not in taken and hits[u][o] > bv:
                    best, bv = o, hits[u][o]
            if best >= 0:
                pairs[u], _ = best, taken.add(best)

    per_unit = {}
    tot_h = tot_t = tot_o = 0
    for u in range(n_units):
        o = pairs.get(u, -1)
        h = hits[u][o] if o >= 0 else 0
        n_t = sum(1 for (_, uu) in truth if uu == u)
        n_o = sum(1 for (_, oo) in preds if oo == o) if o >= 0 else 0
        prec = h / n_o if n_o else 0.0
        rec = h / n_t if n_t else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_unit[u] = {"neuron": o, "hits": h, "n_truth": n_t,
                       "n_out": n_o, "precision": prec, "recall": rec, "f1": f1}
        tot_h += h
        tot_t += n_t
        tot_o += n_o

    global_f = 2 * tot_h / (tot_t + tot_o) if (tot_t + tot_o) else 0.0
    return {"per_unit": per_unit, "hits_matrix": hits,
            "global_f": global_f, "pairs": pairs}


def diagnose(cfg: SorterConfig, signal: Sequence[int],
             truth: Sequence[Tuple[int, int]], n_units: int) -> None:
    """Zwei Messungen, die erklaeren, WORAN die Sortierung haengt.

    D1  Stabilisieren sich die Spikefolgen der Zwischenschicht?
        Die Ausgabeschicht kann nur lernen, was reproduzierbar ist.

    D2  Steckt die Information stattdessen im Ereignisprofil ueber den
        gesamten Attention-Burst? Geprueft mit einem Nearest-Centroid-
        Klassifikator als Obergrenze.
    """
    from collections import Counter, defaultdict

    f = cfg.front
    net = SpikeSorter(cfg)
    ev: List[Tuple[int, int]] = []
    feats: List[Tuple[int, List[int]]] = []
    burst: Optional[List] = None

    for t, x in enumerate(signal):
        r = net.process_sample([x])
        if r.inter >= 0:
            ev.append((t, r.inter))
        if r.attention:
            if burst is None:
                burst = [t, [0] * cfg.n_lines]
            for i, b in enumerate(r.events):
                if b:
                    burst[1][i] += 1
        elif burst is not None:
            feats.append((burst[0], burst[1]))
            burst = None

    # --- D1 ---------------------------------------------------------------
    win = int(0.0025 * f.fs_hz)
    half = len(signal) // 2
    print("\n--- D1: Stabilitaet der Zwischensequenzen ---")
    for label, lo, hi in (("frueh", 0, half), ("spaet", half, len(signal))):
        seqs = defaultdict(list)
        for (t0, u) in truth:
            if lo <= t0 < hi:
                seqs[u].append(tuple(j for (t, j) in ev if t0 - win <= t <= t0 + win))
        parts = []
        for u in sorted(seqs):
            c = Counter(seqs[u])
            n = len(seqs[u])
            top = c.most_common(1)[0][1] if c else 0
            parts.append("Zelle %d: %d/%d eindeutig, haeufigste %.0f%%"
                         % (u, len(c), n, 100 * top / max(1, n)))
        print("  %-6s | %s" % (label, " | ".join(parts)))
    print("  -> Naehert sich 'eindeutig' der AP-Zahl, sind die Folgen NICHT")
    print("     reproduzierbar und die Ausgabeschicht kann nichts lernen.")

    # --- D2 ---------------------------------------------------------------
    win2 = int(0.0030 * f.fs_hz)
    lab = []
    for (t0, vec) in feats:
        m = [u for (t, u) in truth if abs(t - t0) <= win2]
        if m:
            lab.append((vec, m[0]))

    print("\n--- D2: Trennbarkeit des Burst-Ereignisprofils ---")
    if len(lab) < 3 * n_units:
        print("  zu wenige Bursts fuer eine Aussage")
        return
    cent = {}
    for u in range(n_units):
        rows = [v for (v, uu) in lab if uu == u]
        if rows:
            cent[u] = [sum(r[i] for r in rows) / len(rows)
                       for i in range(cfg.n_lines)]
    ok = 0
    for (v, u) in lab:
        d = {c: sum((v[i] - cent[c][i]) ** 2 for i in range(cfg.n_lines))
             for c in cent}
        if min(d, key=d.get) == u:
            ok += 1
    print("  Nearest-Centroid auf dem Burst-Profil: %.1f %% (Zufall %.0f %%)"
          % (100 * ok / len(lab), 100.0 / n_units))
    print("  -> Ist dieser Wert hoch, WAEHREND D1 instabil ist, liegt die")
    print("     Information in der Menge der Leitungen ueber das ganze Muster,")
    print("     nicht in der Tick-Reihenfolge.")


def run_sorter(cfg: SorterConfig, signal: Sequence[int]) -> dict:
    net = SpikeSorter(cfg)
    preds: List[Tuple[int, int]] = []
    inter_spikes = 0
    att_ticks = 0
    for t, x in enumerate(signal):
        r = net.process_sample([x])
        att_ticks += r.attention
        if r.inter >= 0:
            inter_spikes += 1
        if r.output >= 0:
            preds.append((t, r.output))
    return {"net": net, "preds": preds,
            "inter_spikes": inter_spikes, "att_ticks": att_ticks}


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------


def main() -> None:
    cfg = SorterConfig()
    f = cfg.front

    print("=" * 84)
    print("SPIKE-SORTER: Delta-Encoder + Attention + Zwischenschicht + Ausgabeschicht")
    print("=" * 84)
    print("Encoder     : k=%d, %d Leitungen, Rueckblick %.2f ms"
          % (f.k, f.n_lines, f.lookback_ms))
    print("Attention   : Th=%d, w_self=%d, Leck-Shift=%d"
          % (f.v_threshold, f.w_self, f.leak_shift))
    print("Zwischen    : %d Neuronen, Th=%d, w_att=%d, Refrakt=%d Ticks"
          % (cfg.n_inter, cfg.im_threshold, cfg.im_w_attention, cfg.im_refractory))
    print("Verzoegerung: %d Abgriffe x %d Ticks -> bis %.2f ms"
          % (cfg.n_delays, cfg.delay_step, 1000 * cfg.max_delay / f.fs_hz))
    print("Ausgabe     : %d Neuronen, %d Synapsen je Neuron, Th in [%d, %d]"
          % (cfg.n_output, cfg.n_inter * cfg.n_delays, cfg.ol_th_min, cfg.ol_th_max))

    n_units = 3
    signal, truth = make_multiunit_signal(cfg, duration_s=6.0, n_units=n_units,
                                          noise_sigma=25, snr=7.0)
    print("\nTestsignal: %d Samples (%.1f s), %d Aktionspotenziale von %d Zellen"
          % (len(signal), len(signal) / f.fs_hz, len(truth), n_units))

    res = run_sorter(cfg, signal)
    net = res["net"]

    tol_pre = int(0.0005 * f.fs_hz)
    tol_post = int(0.0040 * f.fs_hz)
    sc = match_and_score(truth, res["preds"], n_units, cfg.n_output,
                         tol_pre, tol_post)

    print("\n--- Aktivitaet ---")
    print("Attention aktiv    : %d Ticks (%.2f %% der Zeit)"
          % (res["att_ticks"], 100 * res["att_ticks"] / len(signal)))
    print("Zwischenspikes     : %d (%.1f je Aktionspotenzial)"
          % (res["inter_spikes"], res["inter_spikes"] / max(1, len(truth))))
    print("Ausgabespikes      : %d" % len(res["preds"]))

    print("\n--- Trefferzahlen (Zeile = wahre Zelle, Spalte = Ausgabeneuron) ---")
    print("        " + "".join("  N%d " % o for o in range(cfg.n_output)))
    for u, row in enumerate(sc["hits_matrix"]):
        print("Zelle %d " % u + "".join("%4d " % v for v in row))

    print("\n--- Sortierleistung ---")
    print("Zelle | Neuron | wahr | aus | Treffer | Prec  | Rec   | F1")
    for u in range(n_units):
        d = sc["per_unit"][u]
        print("  %d   |   %2d   | %4d | %3d | %7d | %.3f | %.3f | %.3f"
              % (u, d["neuron"], d["n_truth"], d["n_out"], d["hits"],
                 d["precision"], d["recall"], d["f1"]))
    print("Globaler F-Score: %.3f" % sc["global_f"])

    print("\n--- Gelernte Schwellen der Ausgabeneuronen (Intrinsic Plasticity) ---")
    print("Start war einheitlich %d:" % cfg.ol_th_init)
    for o in range(cfg.n_output):
        used = any(d["neuron"] == o for d in sc["per_unit"].values())
        print("  Neuron %d: Th = %6d   %s"
              % (o, net.out.th[o], "<- hat eine Zelle gelernt" if used else ""))

    diagnose(cfg, signal, truth, n_units)

    print("\n--- Ereignisbasiert vs. dicht (Antwort auf die AER-Frage) ---")
    td = net.inter.op_dense + net.out.op_dense
    ts = net.inter.op_sparse + net.out.op_sparse
    print("Synapsenoperationen dicht        : %10d" % td)
    print("Synapsenoperationen ereignisbasiert: %8d" % ts)
    print("Einsparung                        : %.1f-fach" % (td / max(1, ts)))
    print("\nDazu genuegt ein Priority Encoder auf dem Bitvektor.")
    print("Ein AER-Bus mit Arbiter und Handshake ist dafuer NICHT noetig.")


if __name__ == "__main__":
    main()
