#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITK CIT – VAE Concept-Drift Pipeline (átdolgozott változat)
============================================================

Módosítások az eredeti notebookhoz képest:
  * 10 busz a 3 helyett.
  * stride = 60 (NINCS átfedés az ablakok között).
  * Train/Validation/Test split 70/20/10, rétegezve (TrackID × YearMonth).
  * EarlyStopping CSAK a 20. epoch UTÁN aktív (KL annealing lezárul).
  * Minimum 20, maximum 50 epoch.
  * Ablation: AmbTemp BE / KI – két külön VAE betanítás és összehasonlítás.
  * Két baseline concept-drift / anomáliadetektáló módszer (IsolationForest, PCA-rekonstrukció).
  * Lineáris vs Szinuszos vs Joint (lineáris+szinuszos) modell AIC/BIC összehasonlítással.
  * FPR bootstrap konfidencia-intervallummal, nagyobb mintával, train/test szétválasztással.
  * Minden szöveges eredmény CSV-be, képek külön mappába, futtatásonként új könyvtárba.

Futtatás:
    python itk_cit_vae_pipeline.py

"""

# =============================================================================
# IMPORTOK
# =============================================================================
import os
import sys
import json
import gc
import pickle
import warnings
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless mentéshez
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split

from scipy.optimize import curve_fit
from scipy.spatial.distance import euclidean, mahalanobis
from scipy.stats import f as f_dist, chi2

warnings.filterwarnings("ignore")

# =============================================================================
# KONFIGURÁCIÓ
# =============================================================================
CSV_PATH        = "datas/device_candata.csv"
START_DATE      = "2024-05-01 00:00:00"
END_DATE        = "2025-12-31 23:59:59"
N_BUSES         = 10            # 10 busz
WINDOW_SIZE     = 60            # 60 másodperces ablak
STRIDE          = 60           
LATENT_DIM      = 8
LEARNING_RATE   = 0.0005
BATCH_SIZE      = 2048
MIN_EPOCHS      = 20            # legalább 20 epoch (KL annealing miatt)
MAX_EPOCHS      = 50
KL_ANNEAL_START = 5             # β = 0 az 5. epochig
KL_ANNEAL_END   = 20            # β = 1 a 20. epochig
EARLY_STOP_START = KL_ANNEAL_END + 1   # 1-indexelt: a 21. epoch UTÁN nézzük a loss-t
EARLY_STOP_PATIENCE = 8         # EarlyStopping türelem (a 21. epoch UTÁN)
TRAIN_FRAC      = 0.70
VAL_FRAC        = 0.20
TEST_FRAC       = 0.10
RANDOM_SEED     = 42
N_BOOTSTRAP     = 1000          # FPR bootstrap iterációk száma
TIMESTAMP       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR         = Path("runs") / f"run_{TIMESTAMP}"
PLOTS_DIR       = RUN_DIR / "plots"
CSV_DIR         = RUN_DIR / "csv"
MODELS_DIR      = RUN_DIR / "models"
LOG_PATH        = RUN_DIR / "run_log.txt"

# Mappastruktúra létrehozása
for d in (RUN_DIR, PLOTS_DIR, CSV_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Determinizmus
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    if gpus:
        print(f"GPU(k) elérhető: {[g.name for g in gpus]}")
    else:
        print("GPU nem található, CPU módban futunk.")
except Exception as _e:
    print(f"GPU konfiguráció figyelmen kívül hagyva: {_e}")

# Matplotlib stílus (publikációs minőség)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 110,
    "savefig.dpi": 200,
})

MONTH_NAMES_HU = {1: "Jan", 2: "Feb", 3: "Már", 4: "Ápr", 5: "Máj", 6: "Jún",
                  7: "Júl", 8: "Aug", 9: "Szept", 10: "Okt", 11: "Nov", 12: "Dec"}

# =============================================================================
# LOGGER (a stdout-ot és a run_log.txt-t is kiszolgáljuk)
# =============================================================================
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_log_file = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_file)

def log_section(title):
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")

print(f"Futtatás ID: {TIMESTAMP}")
print(f"Output könyvtár: {RUN_DIR.resolve()}")


# =============================================================================
# 1. ADATBETÖLTÉS ÉS ELŐSZŰRÉS
# =============================================================================
log_section("1. ADATBETÖLTÉS")

if not Path(CSV_PATH).exists():
    raise FileNotFoundError(
        f"Az adatfájl ({CSV_PATH}) nem található. Állítsd be helyesen a CSV_PATH-t a script tetején."
    )

candata = pd.read_csv(CSV_PATH)
print(f"Beolvasott sorok száma: {len(candata):,}")
print(f"Oszlopok: {list(candata.columns)}")

candata["happened_at"] = pd.to_datetime(candata["happened_at"])
print(f"Időszak: {candata['happened_at'].min()}  →  {candata['happened_at'].max()}")


# =============================================================================
# 2. IDŐSZÛRÉS + 10 LEGAKTÍVABB BUSZ
# =============================================================================
log_section("2. IDŐSZÛRÉS + 10 LEGAKTÍVABB BUSZ KIVÁLASZTÁSA")

df_period = candata[(candata["happened_at"] >= START_DATE) &
                    (candata["happened_at"] <= END_DATE)].copy()
print(f"Sorok az időszakban: {len(df_period):,}")

top_buses = df_period["TrackID"].value_counts().head(N_BUSES).index.tolist()
print(f"Top {N_BUSES} busz: {top_buses}")

df_filtered = df_period[df_period["TrackID"].isin(top_buses)].copy()
print(f"Szûrt adatok mérete: {len(df_filtered):,} sor")

del candata, df_period
gc.collect()


# =============================================================================
# 3. OSZLOP-ANALÍZIS ÉS DINAMIKUS SZÛRÉS
# =============================================================================
log_section("3. OSZLOP-ANALÍZIS")

# Hány buszra van adat oszloponként
bus_counts = (df_filtered.groupby("TrackID").count() > 0).sum()

# Megtartjuk azokat az oszlopokat, amelyek legalább a buszok 2/3-ánál léteznek
min_valid_buses = int(np.ceil(N_BUSES * 2 / 3))
valid_meas_columns = [c for c in bus_counts.index
                      if bus_counts[c] >= min_valid_buses
                      and c not in ("happened_at", "TrackID")]
critical_columns = [c for c in bus_counts.index
                    if bus_counts[c] == N_BUSES
                    and c not in ("happened_at", "TrackID")]
print(f"Megtartott mérési oszlopok (>= {min_valid_buses} busz): {valid_meas_columns}")
print(f"Kritikus oszlopok (mind a {N_BUSES} buszra megvan): {critical_columns}")

keep_columns = ["TrackID", "happened_at"] + valid_meas_columns
df_track = df_filtered[keep_columns].copy().sort_values(["TrackID", "happened_at"]).reset_index(drop=True)
del df_filtered
gc.collect()

# Oszlopanalízis riport CSV-be mentése
oszlop_riport = pd.DataFrame({
    "Globális_NaN_pct": df_track[valid_meas_columns].isna().mean() * 100,
    "Buszok_száma_adattal": bus_counts.loc[valid_meas_columns],
    "Összes_busz": N_BUSES,
}).round(2)
oszlop_riport.to_csv(CSV_DIR / "01_oszlop_analizis.csv", index=True)


# =============================================================================
# 4. TRIPID GENERÁLÁS + INTERPOLÁCIÓ 1 Hz-RE
# =============================================================================
log_section("4. TRIPID GENERÁLÁS + 1 Hz RESAMPLING")

time_diff = df_track.groupby("TrackID")["happened_at"].diff()
threshold = pd.Timedelta(minutes=5)
is_new_trip = (time_diff > threshold) | time_diff.isna()
df_track["TripID"] = is_new_trip.cumsum().astype("int32")

# Lyukak kitöltése utazásokon belül
df_track[valid_meas_columns] = df_track.groupby("TripID")[valid_meas_columns].ffill()
df_track = df_track.dropna(subset=critical_columns).fillna(-1)
print(f"Sorok a tisztítás után: {len(df_track):,}, TripID-k: {df_track['TripID'].nunique():,}")

# Memóriaoptimalizálás
float_cols = df_track.select_dtypes(include=["float64"]).columns
df_track[float_cols] = df_track[float_cols].astype("float32")
df_track["TrackID"] = df_track["TrackID"].astype("int32")

df_track = df_track.set_index("happened_at")
feature_cols_all = [c for c in df_track.columns if c not in ("TrackID", "TripID")]

# Resample buszonként (memória-barát)
resampled_chunks = []
unique_buses = df_track["TrackID"].unique()
for i, bus_id in enumerate(unique_buses, 1):
    print(f"  [{i}/{len(unique_buses)}] Busz {bus_id} resampling...")
    bus_chunk = df_track[df_track["TrackID"] == bus_id]
    if bus_chunk.empty:
        continue
    trip_to_track = bus_chunk.groupby("TripID")["TrackID"].first()
    bus_resampled = (bus_chunk.groupby("TripID")[feature_cols_all]
                              .resample("1s").interpolate(method="linear"))
    bus_resampled = bus_resampled.reset_index()
    bus_resampled["TrackID"] = bus_resampled["TripID"].map(trip_to_track)
    bus_resampled[feature_cols_all] = bus_resampled[feature_cols_all].astype("float32")
    bus_resampled["TrackID"] = bus_resampled["TrackID"].astype("int32")
    bus_resampled["TripID"] = bus_resampled["TripID"].astype("int32")
    resampled_chunks.append(bus_resampled)
    del bus_chunk, bus_resampled
    gc.collect()

df_resampled = pd.concat(resampled_chunks, ignore_index=True)
del resampled_chunks, df_track
gc.collect()
print(f"Resampling után: {len(df_resampled):,} sor")


# =============================================================================
# 5. ABLAKOZÁS – stride=60 (NINCS ÁTFEDÉS), kétféle feature-szettre:
#    A) WITH    AmbTemp  (külső hőmérséklet bemenetként)
#    B) WITHOUT AmbTemp  (ablation – ez bizonyítja a tézist)
# =============================================================================
log_section("5. ABLAKOZÁS (stride=60, nincs átfedés)")

def build_windows(df, feature_cols, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Ablakok és metaadataik egyetlen pásztában.
    Visszaad: X (N, window_size, F), track (N,), trip (N,), year_month (N,)
    """
    X_list, track_list, trip_list, ym_list = [], [], [], []
    for trip_id, group in df.groupby("TripID"):
        n = len(group)
        if n < window_size:
            continue
        data = group[feature_cols].to_numpy(dtype="float32")
        track_id = int(group["TrackID"].iloc[0])
        ym_values = group["happened_at"].dt.to_period("M").astype(str).to_numpy()
        for i in range(0, n - window_size + 1, stride):
            X_list.append(data[i:i + window_size])
            track_list.append(track_id)
            trip_list.append(int(trip_id))
            ym_list.append(ym_values[i])
    X = np.asarray(X_list, dtype="float32")
    return X, np.asarray(track_list), np.asarray(trip_list), np.asarray(ym_list)

feature_cols_with    = [c for c in feature_cols_all if c not in ("TrackID", "TripID")]
feature_cols_without = [c for c in feature_cols_with if c != "AmbTemp"]
print(f"Feature-ek (WITH AmbTemp):    {feature_cols_with}")
print(f"Feature-ek (WITHOUT AmbTemp): {feature_cols_without}")

X_with,    track_lbl, trip_lbl, ym_lbl = build_windows(df_resampled, feature_cols_with)
# A WITHOUT verzióhoz NEM kell újra meta -- ugyanazok az ablakok, csak az AmbTemp oszlop nélkül
# Megkeressük az AmbTemp oszlop pozícióját és levágjuk
amb_idx = feature_cols_with.index("AmbTemp") if "AmbTemp" in feature_cols_with else None
if amb_idx is not None:
    X_without = np.delete(X_with, amb_idx, axis=2)
else:
    print("FIGYELEM: nincs AmbTemp oszlop, ablation kihagyva!")
    X_without = X_with.copy()

print(f"Ablakok száma: {len(X_with):,}")
print(f"X_with shape:    {X_with.shape}")
print(f"X_without shape: {X_without.shape}")

# Ablak meta-info mentése CSV-be
ablak_meta = pd.DataFrame({
    "ablak_idx": np.arange(len(X_with)),
    "TrackID": track_lbl,
    "TripID": trip_lbl,
    "YearMonth": ym_lbl,
})
ablak_meta.to_csv(CSV_DIR / "02_ablak_metadata.csv", index=False)

print("\nAblakok eloszlása busz × hónap szerint:")
print(pd.crosstab(track_lbl, ym_lbl))


# =============================================================================
# 6. RÉTEGEZETT TRAIN / VAL / TEST SPLIT (TrackID × YearMonth)
#    A 70/20/10 arány MINDEN (busz, hónap) csoporton belül érvényes.
# =============================================================================
log_section("6. RÉTEGEZETT 70/20/10 SPLIT (TrackID × YearMonth)")

def stratified_three_way_split(track, ym, train_frac, val_frac, test_frac, seed):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    strata = np.array([f"{t}__{m}" for t, m in zip(track, ym)])
    for s in np.unique(strata):
        idxs = np.where(strata == s)[0]
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * train_frac))
        n_val   = int(round(n * val_frac))
        # A maradék a test
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])
    return (np.array(train_idx, dtype=np.int64),
            np.array(val_idx,   dtype=np.int64),
            np.array(test_idx,  dtype=np.int64))

train_idx, val_idx, test_idx = stratified_three_way_split(
    track_lbl, ym_lbl, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, RANDOM_SEED)
print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

# Sanity check: arányok bemutatása CSV-ben
split_lookup = np.empty(len(track_lbl), dtype=object)
split_lookup[train_idx] = "train"
split_lookup[val_idx]   = "val"
split_lookup[test_idx]  = "test"

split_check = pd.DataFrame({
    "TrackID": track_lbl, "YearMonth": ym_lbl, "Split": split_lookup
})
split_table = (split_check.groupby(["TrackID", "YearMonth", "Split"]).size()
                          .unstack(fill_value=0)
                          .reindex(columns=["train", "val", "test"], fill_value=0))
split_table.to_csv(CSV_DIR / "03_split_aranyok.csv")
print("\nSplit arányok ellenőrzése (busz × hónap × split – első 15 sor):")
print(split_table.head(15))


# =============================================================================
# 7. SKÁLÁZÁS (a scaler-t CSAK a train halmazon fit-eljük, leakage-mentesen)
# =============================================================================
log_section("7. STANDARD SCALING (train-only fit)")

def scale_train_val_test(X, train_idx, val_idx, test_idx):
    N, T, F = X.shape
    flat = X.reshape(-1, F)
    # Train rowsok kigyűjtése a fit-hez
    train_rows = np.concatenate([np.arange(i * T, (i + 1) * T) for i in train_idx])
    scaler = StandardScaler()
    scaler.fit(flat[train_rows])
    scaled = scaler.transform(flat).reshape(N, T, F).astype("float32")
    return scaled, scaler

X_with_scaled,    scaler_with    = scale_train_val_test(X_with,    train_idx, val_idx, test_idx)
X_without_scaled, scaler_without = scale_train_val_test(X_without, train_idx, val_idx, test_idx)

X_train_w, X_val_w, X_test_w = X_with_scaled[train_idx], X_with_scaled[val_idx], X_with_scaled[test_idx]
X_train_wo, X_val_wo, X_test_wo = X_without_scaled[train_idx], X_without_scaled[val_idx], X_without_scaled[test_idx]

# Eredeti (nem skálázott) verzió a baseline metódusokhoz
X_with_raw_train = X_with[train_idx]
X_with_raw_test  = X_with[test_idx]

# Mentés
with open(MODELS_DIR / "scaler_with_ambtemp.pkl",    "wb") as f: pickle.dump(scaler_with, f)
with open(MODELS_DIR / "scaler_without_ambtemp.pkl", "wb") as f: pickle.dump(scaler_without, f)

# Feature-statisztikák CSV-be (train halmazon)
feature_stats = pd.DataFrame({
    "Feature":  feature_cols_with,
    "Mean":     scaler_with.mean_,
    "Std":      scaler_with.scale_,
    "Min_train": X_with[train_idx].reshape(-1, X_with.shape[2]).min(axis=0),
    "Max_train": X_with[train_idx].reshape(-1, X_with.shape[2]).max(axis=0),
})
feature_stats.to_csv(CSV_DIR / "04_feature_statisztikak_train.csv", index=False)
print(feature_stats.to_string(index=False))


# =============================================================================
# 8. VAE MODELL (újrahasználható factory)
# =============================================================================
log_section("8. VAE MODELL DEFINIÁLÁSA")

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae(input_shape, latent_dim=LATENT_DIM):
    # ENCODER
    enc_in = keras.Input(shape=input_shape)
    x = layers.Conv1D(32, 3, activation="relu", strides=2, padding="same")(enc_in)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(enc_in, [z_mean, z_log_var, z], name="encoder")

    # DECODER (60 / 4 = 15 a strides miatt)
    dec_in = keras.Input(shape=(latent_dim,))
    x = layers.Dense(15 * 64, activation="relu")(dec_in)
    x = layers.Reshape((15, 64))(x)
    x = layers.Conv1DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv1DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    dec_out = layers.Conv1DTranspose(input_shape[1], 3, activation="linear", padding="same")(x)
    decoder = keras.Model(dec_in, dec_out, name="decoder")

    return encoder, decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = tf.Variable(0.0, trainable=False)
        self.total_loss_tracker  = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker  = keras.metrics.Mean(name="mse")
        self.kl_loss_tracker     = keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs):
        zm, zlv, z = self.encoder(inputs)
        return self.decoder(z)

    def _compute_losses(self, data):
        zm, zlv, z = self.encoder(data)
        recon = self.decoder(z)
        recon_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mse(data, recon), axis=1))
        kl = -0.5 * (1 + zlv - tf.square(zm) - tf.exp(tf.clip_by_value(zlv, -10.0, 10.0)))
        kl = tf.reduce_mean(tf.reduce_sum(kl, axis=1))
        total = recon_loss + self.beta * kl
        return total, recon_loss, kl

    def train_step(self, data):
        with tf.GradientTape() as tape:
            total, recon, kl = self._compute_losses(data)
        grads = tape.gradient(total, self.trainable_weights)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {"loss": self.total_loss_tracker.result(),
                "mse":  self.recon_loss_tracker.result(),
                "kl":   self.kl_loss_tracker.result(),
                "beta": self.beta}

    def test_step(self, data):
        if isinstance(data, tuple): data = data[0]
        total, recon, kl = self._compute_losses(data)
        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {"loss": self.total_loss_tracker.result(),
                "mse":  self.recon_loss_tracker.result(),
                "kl":   self.kl_loss_tracker.result(),
                "beta": self.beta}


# --- Custom callback: KL annealing ---
class KLWeightUpdater(callbacks.Callback):
    def __init__(self, start_epoch=KL_ANNEAL_START, end_epoch=KL_ANNEAL_END):
        super().__init__()
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.start_epoch:
            new_beta = 0.0
        elif epoch >= self.end_epoch:
            new_beta = 1.0
        else:
            new_beta = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        self.model.beta.assign(new_beta)
        print(f"  KL β = {new_beta:.3f}")


# --- Custom callback: EarlyStopping CSAK a start_epoch UTÁN ---
class DelayedEarlyStopping(callbacks.Callback):
    """
    EarlyStopping ami csak start_epoch UTÁN aktív. A KL annealing ideje alatt
    a loss "keveredik" (recon + β*KL, β még változik), ezért nem reális
    konvergencia-jelet ad. Csak a 20. epoch UTÁN nézzük a val_loss-t.
    """
    def __init__(self, monitor="val_loss", patience=EARLY_STOP_PATIENCE,
                 start_epoch=MIN_EPOCHS, min_epochs=MIN_EPOCHS,
                 restore_best_weights=True):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.start_epoch = start_epoch
        self.min_epochs = min_epochs
        self.restore_best_weights = restore_best_weights
        self.best = np.inf
        self.wait = 0
        self.best_weights = None
        self.best_epoch = -1
        self.stopped_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is None:
            return
        # A start_epoch előtt SEMMIT nem csinálunk (sem reset, sem patience, sem mentés).
        # A loss itt KL-vel kontaminált, az EarlyStop és a best-weight mentés
        # is csak az érett (β=1) régióban érvényes.
        if epoch + 1 < self.start_epoch:
            print(f"  [EarlyStop INACTIVE – KL annealing] epoch {epoch+1}, {self.monitor}={current:.4f}")
            return
        if current < self.best:
            self.best = current
            self.wait = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            print(f"  [EarlyStop] epoch {epoch+1}: új legjobb {self.monitor}={current:.4f}")
        else:
            self.wait += 1
            print(f"  [EarlyStop] epoch {epoch+1}: nincs javulás ({self.wait}/{self.patience}), legjobb={self.best:.4f}")
            if self.wait >= self.patience and (epoch + 1) >= self.min_epochs:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                print(f"  >>> Early stopping epoch {epoch+1}-nél. Legjobb epoch: {self.best_epoch+1}, "
                      f"{self.monitor}={self.best:.4f}")


# --- Custom callback: ModelCheckpoint CSAK a start_epoch UTÁN ---
class DelayedCheckpoint(callbacks.Callback):
    def __init__(self, filepath, monitor="val_loss", start_epoch=MIN_EPOCHS):
        super().__init__()
        self.filepath = str(filepath)
        self.monitor = monitor
        self.start_epoch = start_epoch
        self.best = np.inf
    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 < self.start_epoch:
            return
        current = (logs or {}).get(self.monitor)
        if current is None: return
        if current < self.best:
            self.best = current
            self.model.save_weights(self.filepath)
            print(f"  [Checkpoint] epoch {epoch+1}: súlyok mentve ({self.monitor}={current:.4f})")


# =============================================================================
# 9. KÉT MENETBEN BETANÍTÁS (ABLATION: WITH vs WITHOUT AmbTemp)
# =============================================================================
def train_vae_variant(variant_name, X_train, X_val, input_shape, tag):
    """Egy VAE menet betanítása + a history visszaadása."""
    print(f"\n>>> Training variant: {variant_name}, input_shape={input_shape}")
    enc, dec = build_vae(input_shape)
    vae = VAE(enc, dec)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    # tf.data.Dataset – egységes és problémamentes a custom train_step-pel
    train_ds = (tf.data.Dataset.from_tensor_slices(X_train)
                  .shuffle(min(20000, len(X_train)), seed=RANDOM_SEED, reshuffle_each_iteration=True)
                  .batch(BATCH_SIZE)
                  .prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices(X_val)
                  .batch(BATCH_SIZE)
                  .prefetch(tf.data.AUTOTUNE))

    ckpt_path = MODELS_DIR / f"best_vae_{tag}.weights.h5"
    es = DelayedEarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE,
                              start_epoch=EARLY_STOP_START, min_epochs=MIN_EPOCHS,
                              restore_best_weights=True)
    cb = [
        KLWeightUpdater(KL_ANNEAL_START, KL_ANNEAL_END),
        DelayedCheckpoint(ckpt_path, monitor="val_loss", start_epoch=EARLY_STOP_START),
        es,
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7,
                                    cooldown=3, min_lr=1e-6, verbose=0),
    ]
    history = vae.fit(
        train_ds, validation_data=val_ds,
        epochs=MAX_EPOCHS, verbose=2, callbacks=cb,
    )
    # Tényleges futtatott epochok száma
    actual_epochs = len(history.history["loss"])
    print(f"  Tényleges futott epochok: {actual_epochs} (best epoch: {es.best_epoch+1 if es.best_epoch>=0 else '–'})")

    # Mentés (full súly + json)
    vae.save_weights(MODELS_DIR / f"vae_{tag}_final.weights.h5")
    enc.save(MODELS_DIR / f"encoder_{tag}.keras")
    dec.save(MODELS_DIR / f"decoder_{tag}.keras")

    return vae, enc, dec, history, actual_epochs, es.best_epoch + 1 if es.best_epoch >= 0 else -1


log_section("9. VAE TANÍTÁS – WITH AmbTemp")
vae_w, enc_w, dec_w, hist_w, n_ep_w, best_ep_w = train_vae_variant(
    "WITH AmbTemp", X_train_w, X_val_w, (WINDOW_SIZE, X_train_w.shape[2]), "with"
)

log_section("9b. VAE TANÍTÁS – WITHOUT AmbTemp (ABLATION)")
vae_wo, enc_wo, dec_wo, hist_wo, n_ep_wo, best_ep_wo = train_vae_variant(
    "WITHOUT AmbTemp", X_train_wo, X_val_wo, (WINDOW_SIZE, X_train_wo.shape[2]), "without"
)

# History CSV-be
for tag, hist in (("with", hist_w), ("without", hist_wo)):
    hdf = pd.DataFrame(hist.history)
    hdf.index.name = "epoch"
    # A tf.Variable béta értékek konverziója
    if "beta" in hdf.columns:
        hdf["beta"] = hdf["beta"].apply(lambda x: float(x) if hasattr(x, "__float__") else x)
    hdf.to_csv(CSV_DIR / f"05_training_history_{tag}.csv")


# Tanítási görbék összehasonlító plot
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for row, (tag, hist) in enumerate((("with", hist_w), ("without", hist_wo))):
    h = hist.history
    axes[row, 0].plot(h["loss"], label="train loss")
    axes[row, 0].plot(h["val_loss"], label="val loss")
    axes[row, 0].axvline(MIN_EPOCHS - 1, color="red", ls=":", alpha=0.7, label="KL ann. vége (epoch 20)")
    axes[row, 0].set_title(f"Total Loss ({tag} AmbTemp)")
    axes[row, 0].set_xlabel("Epoch"); axes[row, 0].set_ylabel("Loss"); axes[row, 0].legend(); axes[row, 0].grid(alpha=.3)

    axes[row, 1].plot(h["mse"], label="train MSE")
    axes[row, 1].plot(h["val_mse"], label="val MSE")
    axes[row, 1].axvline(MIN_EPOCHS - 1, color="red", ls=":", alpha=0.7)
    axes[row, 1].set_title(f"Reconstruction MSE ({tag})")
    axes[row, 1].set_xlabel("Epoch"); axes[row, 1].legend(); axes[row, 1].grid(alpha=.3)

    axes[row, 2].plot(h["kl"], label="train KL")
    axes[row, 2].plot(h["val_kl"], label="val KL")
    axes[row, 2].axvline(MIN_EPOCHS - 1, color="red", ls=":", alpha=0.7)
    axes[row, 2].set_title(f"KL Divergence ({tag})")
    axes[row, 2].set_xlabel("Epoch"); axes[row, 2].legend(); axes[row, 2].grid(alpha=.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_training_curves.png", bbox_inches="tight")
plt.close()


# =============================================================================
# 10. LÁTENS REPREZENTÁCIÓK A TEST HALMAZON (mindkét variánsra)
# =============================================================================
log_section("10. TEST LÁTENS REPREZENTÁCIÓK")

def encode_z(encoder, X):
    zm, _, _ = encoder.predict(X, batch_size=512, verbose=0)
    return zm

z_test_w  = encode_z(enc_w,  X_test_w)
z_test_wo = encode_z(enc_wo, X_test_wo)

# Test meta
track_test = track_lbl[test_idx]
ym_test    = ym_lbl[test_idx]

# Hónap (1-12) és Év-Hónap kinyerése
def parse_ym(ym_str):
    y, m = ym_str.split("-")
    return int(y), int(m)
years_test  = np.array([parse_ym(y)[0] for y in ym_test])
months_test = np.array([parse_ym(y)[1] for y in ym_test])

print(f"Test latens shape (with): {z_test_w.shape}")
print(f"Test latens shape (without): {z_test_wo.shape}")

# Mentés CSV-be
pd.DataFrame(z_test_w,  columns=[f"z_with_{i}"    for i in range(LATENT_DIM)]).to_csv(
    CSV_DIR / "06_test_latents_with.csv", index=False)
pd.DataFrame(z_test_wo, columns=[f"z_without_{i}" for i in range(LATENT_DIM)]).to_csv(
    CSV_DIR / "06_test_latents_without.csv", index=False)


# =============================================================================
# 11. HAVI CENTROID-TÁVOLSÁGOK + SZINUSZOS ILLESZTÉS (TEST HALMAZ)
# =============================================================================
log_section("11. CENTROID-TÁVOLSÁGOK + SZINUSZOS ILLESZTÉS")

def sine_func(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

def linear_func(x, m, b):
    return m * x + b

def joint_func(x, m, b, a, phi):
    """Lineáris + szinuszos, 12 hónapos fix periódussal."""
    return m * x + b + a * np.sin(2 * np.pi * x / 12 + phi)

# Globális Év-Hónap tengely
global_ym = sorted(set(ym_test.tolist()))
ym_to_idx = {ym: i for i, ym in enumerate(global_ym)}
tick_labels = [f"{MONTH_NAMES_HU[int(ym.split('-')[1])]} '{ym.split('-')[0][2:]}" for ym in global_ym]

def per_bus_centroids_and_distances(z, track_lbl, ym_lbl):
    """Buszonként visszaadja az x-indexeket, távolságokat, sine fit-et."""
    bus_results = {}
    for bus in np.unique(track_lbl):
        mask = track_lbl == bus
        unique_ym_bus = sorted(set(ym_lbl[mask].tolist()))
        if len(unique_ym_bus) < 4:
            continue
        centroids = {ym: z[(ym_lbl == ym) & mask].mean(axis=0) for ym in unique_ym_bus}
        ref = centroids[unique_ym_bus[0]]
        x = np.array([ym_to_idx[ym] for ym in unique_ym_bus], dtype=float)
        y = np.array([euclidean(centroids[ym], ref) for ym in unique_ym_bus])
        bus_results[bus] = {"ym": unique_ym_bus, "x": x, "y": y, "centroids": centroids}
    return bus_results

results_w  = per_bus_centroids_and_distances(z_test_w,  track_test, ym_test)
results_wo = per_bus_centroids_and_distances(z_test_wo, track_test, ym_test)


# AIC/BIC számítás least-squares modellekhez
def aic_bic(rss, n, k):
    """Gauss-zaj feltételezése mellett, max-likelihood."""
    if rss <= 0 or n <= k:
        return np.nan, np.nan
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return aic, bic

def fit_all_models(x, y):
    out = {}
    n = len(y)
    # 1. Lineáris (k=2)
    try:
        popt_lin, _ = curve_fit(linear_func, x, y)
        y_lin = linear_func(x, *popt_lin)
        rss_lin = np.sum((y - y_lin) ** 2)
        r2_lin = r2_score(y, y_lin)
        aic_l, bic_l = aic_bic(rss_lin, n, 2)
        out["linear"] = {"params": popt_lin.tolist(), "rss": rss_lin, "r2": r2_lin,
                         "aic": aic_l, "bic": bic_l, "k": 2}
    except Exception as e:
        out["linear"] = {"error": str(e)}
    # 2. Szinuszos (k=4)
    try:
        a0 = (y.max() - y.min()) / 2
        d0 = y.mean()
        p0 = [a0, 2 * np.pi / 12, 0.0, d0]
        popt_s, _ = curve_fit(sine_func, x, y, p0=p0, maxfev=10000,
                              bounds=([0, 0.4, -np.inf, 0], [np.inf, 0.7, np.inf, np.inf]))
        y_s = sine_func(x, *popt_s)
        rss_s = np.sum((y - y_s) ** 2)
        r2_s = r2_score(y, y_s)
        aic_s, bic_s = aic_bic(rss_s, n, 4)
        out["sinusoidal"] = {"params": popt_s.tolist(), "rss": rss_s, "r2": r2_s,
                             "aic": aic_s, "bic": bic_s, "k": 4}
    except Exception as e:
        out["sinusoidal"] = {"error": str(e)}
    # 3. Joint (lineáris + sinus, 12 hó periódus, k=4)
    try:
        a0 = (y.max() - y.min()) / 2
        p0 = [0.0, y[0], a0, 0.0]
        popt_j, _ = curve_fit(joint_func, x, y, p0=p0, maxfev=10000)
        y_j = joint_func(x, *popt_j)
        rss_j = np.sum((y - y_j) ** 2)
        r2_j = r2_score(y, y_j)
        aic_j, bic_j = aic_bic(rss_j, n, 4)
        out["joint"] = {"params": popt_j.tolist(), "rss": rss_j, "r2": r2_j,
                        "aic": aic_j, "bic": bic_j, "k": 4}
    except Exception as e:
        out["joint"] = {"error": str(e)}
    return out


# Fittelés és vizualizáció buszonként mindkét variánsra
def plot_and_save_centroid_curves(results, tag, model_comparison_rows):
    fig, ax = plt.subplots(figsize=(13, 6))
    cmap = plt.get_cmap("tab10")
    distances_csv_rows = []
    sine_csv_rows = []
    for i, (bus, r) in enumerate(sorted(results.items())):
        color = cmap(i % 10)
        ax.plot(r["x"], r["y"], marker="o", lw=1.6, color=color, label=f"Bus {bus} (mért)")
        models = fit_all_models(r["x"], r["y"])

        # Plotoljuk a JOINT és a SINUSOIDAL trendet (vékony szaggatott)
        if "joint" in models and "params" in models["joint"]:
            x_fit = np.linspace(r["x"].min(), r["x"].max(), 200)
            y_fit = joint_func(x_fit, *models["joint"]["params"])
            ax.plot(x_fit, y_fit, ls="--", lw=1.0, color=color, alpha=0.5)

        # Csv sorok
        for xi, yi, ymv in zip(r["x"], r["y"], r["ym"]):
            distances_csv_rows.append({"TrackID": bus, "YearMonth": ymv,
                                        "x_idx": int(xi), "distance": float(yi),
                                        "variant": tag})
        for mtype, m in models.items():
            row = {"TrackID": bus, "variant": tag, "model": mtype}
            row.update({k: v for k, v in m.items() if k not in ("params",)})
            if "params" in m:
                for j, p in enumerate(m["params"]):
                    row[f"param_{j}"] = p
            model_comparison_rows.append(row)

        # Külön szinuszos-csak sor (kompatibilitás a régi riporttal)
        if "sinusoidal" in models and "params" in models["sinusoidal"]:
            ps = models["sinusoidal"]["params"]
            sine_csv_rows.append({"TrackID": bus, "variant": tag,
                                  "a": ps[0], "b": ps[1], "c": ps[2], "d": ps[3],
                                  "r2": models["sinusoidal"]["r2"]})

    ax.set_xticks(range(len(global_ym)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_title(f"Havi centroid-távolságok ({tag} AmbTemp) – test halmaz, buszonként\n"
                 f"Vékony szaggatott = joint (lineáris+sinus, 12hó periódus)", fontsize=11)
    ax.set_xlabel("Idő (Év-Hónap)")
    ax.set_ylabel("Euklideszi távolság a busz saját kiinduló centroidjától")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"11_centroid_distances_{tag}.png", bbox_inches="tight")
    plt.close()

    pd.DataFrame(distances_csv_rows).to_csv(CSV_DIR / f"07_centroid_distances_{tag}.csv", index=False)
    pd.DataFrame(sine_csv_rows).to_csv(CSV_DIR / f"08_sine_fit_params_{tag}.csv", index=False)


model_comparison_rows = []
plot_and_save_centroid_curves(results_w,  "with",    model_comparison_rows)
plot_and_save_centroid_curves(results_wo, "without", model_comparison_rows)

model_comp_df = pd.DataFrame(model_comparison_rows)
model_comp_df.to_csv(CSV_DIR / "09_model_comparison_aic_bic.csv", index=False)
print("\nModel-összehasonlítás (AIC/BIC) első 12 sora:")
print(model_comp_df[["TrackID", "variant", "model", "r2", "rss", "aic", "bic"]].head(12).to_string(index=False))


# AIC/BIC összegző: átlag buszokra mindkét variánsban
log_section("11b. AIC/BIC ÖSSZEGZÉS")
summary_aic_bic = (model_comp_df.dropna(subset=["aic"])
                    .groupby(["variant", "model"])[["aic", "bic", "r2"]]
                    .agg(["mean", "std", "min", "max"])
                    .round(3))
print(summary_aic_bic.to_string())
summary_aic_bic.to_csv(CSV_DIR / "10_aic_bic_summary.csv")


# Globális (busz-átlagolt) centroid távolság – PCA-vizualizációhoz is
def global_monthly_distances(z, ym_lbl):
    unique_ym = sorted(set(ym_lbl.tolist()))
    centroids = {ym: z[ym_lbl == ym].mean(axis=0) for ym in unique_ym}
    ref = centroids[unique_ym[0]]
    x = np.array([ym_to_idx[ym] for ym in unique_ym], dtype=float)
    y = np.array([euclidean(centroids[ym], ref) for ym in unique_ym])
    return unique_ym, x, y, centroids

unique_ym_w,  x_glob_w,  d_glob_w,  centroids_w  = global_monthly_distances(z_test_w,  ym_test)
unique_ym_wo, x_glob_wo, d_glob_wo, centroids_wo = global_monthly_distances(z_test_wo, ym_test)

global_models_w  = fit_all_models(x_glob_w,  d_glob_w)
global_models_wo = fit_all_models(x_glob_wo, d_glob_wo)

# Plot a globális távolságokról + három modell
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, x, y, models, tag in ((axes[0], x_glob_w,  d_glob_w,  global_models_w,  "with"),
                              (axes[1], x_glob_wo, d_glob_wo, global_models_wo, "without")):
    ax.plot(x, y, "o-", lw=2, color="#1E293B", label="mért")
    x_fit = np.linspace(x.min(), x.max(), 200)
    if "linear" in models and "params" in models["linear"]:
        ax.plot(x_fit, linear_func(x_fit, *models["linear"]["params"]),
                ":", color="orange", label=f"lineáris (AIC={models['linear']['aic']:.1f})")
    if "sinusoidal" in models and "params" in models["sinusoidal"]:
        ax.plot(x_fit, sine_func(x_fit, *models["sinusoidal"]["params"]),
                "--", color="green", label=f"szinuszos (AIC={models['sinusoidal']['aic']:.1f})")
    if "joint" in models and "params" in models["joint"]:
        ax.plot(x_fit, joint_func(x_fit, *models["joint"]["params"]),
                "-.", color="red", label=f"joint (AIC={models['joint']['aic']:.1f})")
    ax.set_xticks(range(len(global_ym)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Globális centroid-távolság – {tag} AmbTemp")
    ax.set_xlabel("Hónap"); ax.set_ylabel("Euklideszi távolság")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "11b_global_centroid_models.png", bbox_inches="tight")
plt.close()


# =============================================================================
# 12. PCA VIZUALIZÁCIÓ – TEST HALMAZ HÓNAPOK SZERINT
# =============================================================================
log_section("12. PCA VIZUALIZÁCIÓ (test halmaz, hónapok)")

def pca_scatter_per_month(z, months, tag):
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z)
    plt.figure(figsize=(11, 8))
    palette = sns.color_palette("husl", 12)
    for m in sorted(np.unique(months)):
        mask = months == m
        plt.scatter(z_pca[mask, 0], z_pca[mask, 1], s=10, alpha=0.5,
                    color=palette[m - 1], label=MONTH_NAMES_HU[m])
    plt.title(f"VAE látens tér (PCA 2D) – {tag} AmbTemp, teszt halmaz\n"
              f"variancia: PC1={pca.explained_variance_ratio_[0]:.1%}, "
              f"PC2={pca.explained_variance_ratio_[1]:.1%}")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(title="Hónap", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"12_pca_months_{tag}.png", bbox_inches="tight")
    plt.close()
    return z_pca, pca

z_pca_w,  pca_w  = pca_scatter_per_month(z_test_w,  months_test, "with")
z_pca_wo, pca_wo = pca_scatter_per_month(z_test_wo, months_test, "without")


# =============================================================================
# 13. HOTELLING T² + MAHALANOBIS (Jan vs Júl, mindkét variánsban)
# =============================================================================
log_section("13. HOTELLING T² (Jan vs Júl)")

def hotelling_test(z_jan, z_jul):
    mu1, mu2 = z_jan.mean(axis=0), z_jul.mean(axis=0)
    cov1, cov2 = np.cov(z_jan, rowvar=False), np.cov(z_jul, rowvar=False)
    n1, n2 = len(z_jan), len(z_jul)
    p = z_jan.shape[1]
    pooled = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)
    inv_pooled = np.linalg.pinv(pooled)
    diff = mu1 - mu2
    t2 = (n1 * n2) / (n1 + n2) * diff @ inv_pooled @ diff.T
    f_stat = t2 * (n1 + n2 - p - 1) / (p * (n1 + n2 - 2))
    p_val = 1 - f_dist.cdf(f_stat, p, n1 + n2 - p - 1)
    # Mahalanobis a két átlag között (az 1. eloszlás cov-jával)
    inv_cov1 = np.linalg.pinv(cov1)
    d_maha = float(mahalanobis(mu1, mu2, inv_cov1))
    return {"T2": float(t2), "F": float(f_stat), "p_value": float(p_val),
            "n_jan": n1, "n_jul": n2, "p_dim": p, "mahalanobis": d_maha}

hot_rows = []
for tag, z in (("with", z_test_w), ("without", z_test_wo)):
    z_jan = z[months_test == 1]
    z_jul = z[months_test == 7]
    if len(z_jan) < LATENT_DIM + 2 or len(z_jul) < LATENT_DIM + 2:
        print(f"  {tag}: kevés Jan ({len(z_jan)}) vagy Júl ({len(z_jul)}) test minta")
        continue
    res = hotelling_test(z_jan, z_jul)
    res["variant"] = tag
    hot_rows.append(res)
    print(f"  {tag}: T²={res['T2']:.2f}, F={res['F']:.2f}, p={res['p_value']:.2e}, "
          f"Mahalanobis={res['mahalanobis']:.3f}, n_jan={res['n_jan']}, n_jul={res['n_jul']}")
pd.DataFrame(hot_rows).to_csv(CSV_DIR / "11_hotelling_t2.csv", index=False)


# =============================================================================
# 14. FALSE-POSITIVE-RATE + BOOTSTRAP CI (cirkularitás-mentesen)
#     - Mahalanobis küszöb a TRAIN HALMAZ januári adatából (referencia)
#     - "Jó" júliusi minták kiválasztása a TRAIN júliusi modelljével
#     - FPR mérés CSAK a TEST júliusi mintákon
#     - Bootstrap 95% CI
# =============================================================================
log_section("14. FPR BOOTSTRAP CI (cirkularitás-mentes)")

# Train-test címkék részére is kell a hónap
ym_train = ym_lbl[train_idx]
months_train = np.array([parse_ym(y)[1] for y in ym_train])

def fpr_bootstrap(z_train, z_test_pts, months_train_arr, months_test_arr, dim,
                  n_boot=N_BOOTSTRAP, alpha=0.05):
    """
    FPR: a TEST halmaz júliusi 'normál' pontjaiból mennyi tűnik anomáliának
    a TRAIN halmazon tanult januári referenciaeloszlás szempontjából.
    Cirkularitás-mentes: a januári mu/cov train-ből jön, a júliusi minták
    "normál"-ságát train júliusi eloszlással szűrjük, az FPR-t test júliusin mérjük.
    """
    z_train_jan = z_train[months_train_arr == 1]
    z_train_jul = z_train[months_train_arr == 7]
    z_test_jul  = z_test_pts[months_test_arr == 7]

    if len(z_train_jan) < dim + 2 or len(z_train_jul) < dim + 2 or len(z_test_jul) < dim + 2:
        return None

    mu_jan = z_train_jan.mean(axis=0)
    cov_jan = np.cov(z_train_jan, rowvar=False)
    inv_jan = np.linalg.pinv(cov_jan)

    mu_jul = z_train_jul.mean(axis=0)
    cov_jul = np.cov(z_train_jul, rowvar=False)
    inv_jul = np.linalg.pinv(cov_jul)

    threshold = chi2.ppf(1 - alpha, dim)

    # "Jó" júliusi test minták – júliusi modellen belül van
    d2_jul_self = np.array([mahalanobis(z, mu_jul, inv_jul) ** 2 for z in z_test_jul])
    good_mask = d2_jul_self <= threshold
    z_good = z_test_jul[good_mask]

    if len(z_good) < 30:
        return {"fpr_point": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "n_good_test_jul": int(len(z_good)),
                "n_train_jan": int(len(z_train_jan)), "n_train_jul": int(len(z_train_jul))}

    # Mennyi mutat anomáliát a JANUÁRI referenciához képest?
    d2_jan = np.array([mahalanobis(z, mu_jan, inv_jan) ** 2 for z in z_good])
    fpr_point = float(np.mean(d2_jan > threshold))

    # Bootstrap CI
    rng = np.random.RandomState(RANDOM_SEED)
    fprs = []
    for _ in range(n_boot):
        boot = rng.choice(len(z_good), size=len(z_good), replace=True)
        d2_boot = d2_jan[boot]
        fprs.append(np.mean(d2_boot > threshold))
    ci_low, ci_high = np.percentile(fprs, [2.5, 97.5])
    return {
        "fpr_point": fpr_point, "ci_low": float(ci_low), "ci_high": float(ci_high),
        "n_good_test_jul": int(len(z_good)),
        "n_train_jan": int(len(z_train_jan)),
        "n_train_jul": int(len(z_train_jul)),
        "threshold_chi2_95": float(threshold),
    }

fpr_rows = []
for tag, z_train, z_test in (("with", encode_z(enc_w,  X_train_w),  z_test_w),
                              ("without", encode_z(enc_wo, X_train_wo), z_test_wo)):
    res = fpr_bootstrap(z_train, z_test, months_train, months_test, LATENT_DIM)
    if res is not None:
        res["variant"] = tag
        fpr_rows.append(res)
        print(f"  {tag}: FPR={res['fpr_point']:.4f} (95% CI [{res['ci_low']:.4f}, {res['ci_high']:.4f}]), "
              f"n_test_jul_good={res['n_good_test_jul']}, "
              f"n_train_jan={res['n_train_jan']}, n_train_jul={res['n_train_jul']}")

pd.DataFrame(fpr_rows).to_csv(CSV_DIR / "12_fpr_bootstrap.csv", index=False)


# =============================================================================
# 15. BASELINE COMPARISON
#     Két alternatív koncepció-eltolódás / anomália módszer:
#       (A) IsolationForest a per-ablak feature-átlagokon
#       (B) PCA-rekonstrukciós hiba (lineáris autoencoder ekvivalens)
#       (C) Bónusz: nyers feature-térben végzett havi-centroid távolság (kontroll)
# =============================================================================
log_section("15. BASELINE COMPARISON (IsolationForest, PCA, raw centroidok)")

# Mindegyiket a TEST halmazon számoljuk, ugyanazokon az ablakokon
X_test_for_baseline = X_with[test_idx]   # nem skálázott eredeti adat
# Window-szintű feature: minden szenzor átlaga az ablakban
X_test_winmean = X_test_for_baseline.mean(axis=1)            # (N_test, F)
X_train_for_baseline = X_with[train_idx]
X_train_winmean = X_train_for_baseline.mean(axis=1)

# --- (A) IsolationForest, TRAIN-en illesztve, TEST-en pontozva
print("  (A) IsolationForest...")
iso = IsolationForest(n_estimators=200, contamination="auto", random_state=RANDOM_SEED)
iso.fit(X_train_winmean)
iso_scores_test = -iso.score_samples(X_test_winmean)  # nagyobb = anomálisabb

# Havi átlag IsolationForest-pontszám (drift-indikátor)
iso_monthly = pd.DataFrame({
    "YearMonth": ym_test, "iso_score": iso_scores_test
}).groupby("YearMonth")["iso_score"].mean().reindex(global_ym)

# --- (B) PCA rekonstrukciós hiba (lineáris baseline)
print("  (B) PCA rekonstrukciós hiba (lineáris dim. csökkentés)...")
X_train_flat = X_train_w.reshape(len(X_train_w), -1)
X_test_flat  = X_test_w.reshape(len(X_test_w),  -1)
pca_base = PCA(n_components=LATENT_DIM, random_state=RANDOM_SEED)
pca_base.fit(X_train_flat)
X_test_proj  = pca_base.transform(X_test_flat)
X_test_recon = pca_base.inverse_transform(X_test_proj)
pca_recon_err = np.mean((X_test_flat - X_test_recon) ** 2, axis=1)
pca_recon_monthly = pd.DataFrame({
    "YearMonth": ym_test, "pca_recon_err": pca_recon_err
}).groupby("YearMonth")["pca_recon_err"].mean().reindex(global_ym)

# --- (C) Nyers feature-tér havi centroid (kontroll)
print("  (C) Raw feature havi centroid távolság...")
raw_centroids = {ym: X_test_winmean[ym_test == ym].mean(axis=0)
                 for ym in global_ym if (ym_test == ym).sum() > 0}
raw_ref = raw_centroids[global_ym[0]]
raw_dist = pd.Series({ym: euclidean(raw_centroids[ym], raw_ref) for ym in raw_centroids},
                     name="raw_distance").reindex(global_ym)

# --- VAE havi-centroid távolság (with) referenciaként
vae_dist_w  = pd.Series(d_glob_w,  index=unique_ym_w,  name="vae_with_dist").reindex(global_ym)
vae_dist_wo = pd.Series(d_glob_wo, index=unique_ym_wo, name="vae_without_dist").reindex(global_ym)

baseline_df = pd.concat([vae_dist_w, vae_dist_wo, iso_monthly, pca_recon_monthly, raw_dist], axis=1)
baseline_df.index.name = "YearMonth"
baseline_df.to_csv(CSV_DIR / "13_baseline_comparison.csv")
print("\n  Havi átlagok (drift-indikátorok):")
print(baseline_df.round(4))

# Mindegyik módszerre AIC/BIC fit (lineáris/sinus/joint)
baseline_models_rows = []
for col in baseline_df.columns:
    y = baseline_df[col].to_numpy()
    if np.isnan(y).any():
        continue
    x = np.arange(len(y), dtype=float)
    models = fit_all_models(x, y)
    for mtype, m in models.items():
        row = {"method": col, "model": mtype}
        row.update({k: v for k, v in m.items() if k != "params"})
        if "params" in m:
            for j, p in enumerate(m["params"]):
                row[f"param_{j}"] = p
        baseline_models_rows.append(row)
pd.DataFrame(baseline_models_rows).to_csv(CSV_DIR / "14_baseline_aic_bic.csv", index=False)

# Baseline vizualizáció (normalizálva, hogy összehasonlítható legyen)
fig, ax = plt.subplots(figsize=(12, 6))
for col in baseline_df.columns:
    y = baseline_df[col]
    if y.notna().sum() < 2: continue
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    ax.plot(range(len(y_norm)), y_norm.values, marker="o", lw=1.5, label=col)
ax.set_xticks(range(len(global_ym)))
ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
ax.set_title("Baseline és VAE koncepció-eltolódás detektorok – havi profil (min-max normalizált)")
ax.set_xlabel("Hónap"); ax.set_ylabel("Normalizált drift-indikátor (0..1)")
ax.grid(alpha=0.3); ax.legend(frameon=False, loc="best")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "13_baseline_comparison.png", bbox_inches="tight")
plt.close()


# =============================================================================
# 16. SZEZONALITÁS-PREDIKCIÓ a TEST HALMAZ LÁTENS VEKTORAIN
#     - Hónap (1-12) és Évszak osztályozás
#     - Belső 80/20 split a test halmazon belül
# =============================================================================
log_section("16. SZEZONALITÁS-PREDIKCIÓ (test halmaz)")

def get_season(month):
    if month in (12, 1, 2): return 0
    if month in (3, 4, 5):  return 1
    if month in (6, 7, 8):  return 2
    return 3

season_test = np.array([get_season(m) for m in months_test])
season_names = ["Tél", "Tavasz", "Nyár", "Ősz"]

def run_classification(z, y, label_names, prefix, tag):
    # Belső split a test halmazon belül
    Xtr, Xte, ytr, yte = train_test_split(z, y, test_size=0.20,
                                          random_state=RANDOM_SEED, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                 random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xte)
    report = classification_report(yte, y_pred, target_names=label_names,
                                   output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(report).T.round(4)
    rep_df.to_csv(CSV_DIR / f"15_classification_{prefix}_{tag}.csv")
    print(f"\n  {prefix} ({tag}):")
    print(rep_df.to_string())

    cm = confusion_matrix(yte, y_pred, labels=list(range(len(label_names))))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f"Confusion Matrix – {prefix} ({tag} AmbTemp)")
    plt.ylabel("Valódi"); plt.xlabel("Megjósolt")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"16_cm_{prefix}_{tag}.png", bbox_inches="tight")
    plt.close()
    return clf, rep_df

# Évszak
clf_season_w,  _ = run_classification(z_test_w,  season_test, season_names, "season", "with")
clf_season_wo, _ = run_classification(z_test_wo, season_test, season_names, "season", "without")

# Hónap (csak ott, ahol van elegendő mintánk hónaponként)
month_unique_test = sorted(np.unique(months_test))
month_label_names = [MONTH_NAMES_HU[m] for m in month_unique_test]
# Re-index 0..k-1
month_to_idx = {m: i for i, m in enumerate(month_unique_test)}
months_test_reidx = np.array([month_to_idx[m] for m in months_test])

if len(month_unique_test) >= 3:
    run_classification(z_test_w,  months_test_reidx, month_label_names, "month", "with")
    run_classification(z_test_wo, months_test_reidx, month_label_names, "month", "without")
else:
    print(f"  Túl kevés hónap ({len(month_unique_test)}) a havi osztályozáshoz – kihagyva.")


# =============================================================================
# 17. ÖSSZEGZŐ JSON
# =============================================================================
log_section("17. RUN SUMMARY MENTÉSE")

summary = {
    "run_id": TIMESTAMP,
    "config": {
        "csv_path": CSV_PATH,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "n_buses": N_BUSES,
        "buses_selected": [int(b) for b in top_buses],
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "latent_dim": LATENT_DIM,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "min_epochs": MIN_EPOCHS,
        "max_epochs": MAX_EPOCHS,
        "kl_anneal_start": KL_ANNEAL_START,
        "kl_anneal_end": KL_ANNEAL_END,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "n_bootstrap": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
    },
    "data": {
        "n_windows_total": int(len(X_with)),
        "n_train": int(len(train_idx)),
        "n_val":   int(len(val_idx)),
        "n_test":  int(len(test_idx)),
        "features_with":    feature_cols_with,
        "features_without": feature_cols_without,
        "ym_values_in_test": global_ym,
    },
    "training": {
        "with_ambtemp":    {"epochs_run": int(n_ep_w),  "best_epoch": int(best_ep_w)},
        "without_ambtemp": {"epochs_run": int(n_ep_wo), "best_epoch": int(best_ep_wo)},
    },
    "hotelling_t2": hot_rows,
    "fpr_bootstrap": fpr_rows,
    "global_aic_bic_with":    {k: ({kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                                    for kk, vv in v.items() if kk != "params"} | {"params": v.get("params")})
                               for k, v in global_models_w.items()},
    "global_aic_bic_without": {k: ({kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                                    for kk, vv in v.items() if kk != "params"} | {"params": v.get("params")})
                               for k, v in global_models_wo.items()},
}

with open(RUN_DIR / "run_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

print(f"\nSummary mentve: {RUN_DIR / 'run_summary.json'}")
print(f"\n=== KÉSZ ===")
print(f"Minden output a következő mappában: {RUN_DIR.resolve()}")
print(f"  - plots/  – {len(list(PLOTS_DIR.glob('*.png')))} PNG")
print(f"  - csv/    – {len(list(CSV_DIR.glob('*.csv')))} CSV")
print(f"  - models/ – {len(list(MODELS_DIR.glob('*')))} fájl")
print(f"  - run_log.txt, run_summary.json")

# Logfájl lezárása
sys.stdout = sys.__stdout__
_log_file.close()