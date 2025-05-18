import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
import torch
from typing import Callable

# --- VAE custom layers & builder ---
@tf.keras.utils.register_keras_serializable()
def sampling(args):
    mean, log_var = args
    eps = K.random_normal(tf.shape(mean))
    return mean + tf.exp(0.5 * log_var) * eps

@tf.keras.utils.register_keras_serializable(package="Custom")
class VAELossLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        orig, recon, mean, log_var = inputs
        recon_loss = tf.reduce_sum(tf.square(orig - recon), axis=1)
        kl_loss    = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
        self.add_loss(tf.reduce_mean(recon_loss + kl_loss))
        return recon


def build_vae(input_dim: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation="relu")(inp)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    z_mean = tf.keras.layers.Dense(4, name="z_mean")(x)
    z_logvar = tf.keras.layers.Dense(4, name="z_log_var")(x)
    z = tf.keras.layers.Lambda(sampling, name="z")([z_mean, z_logvar])

    latent = tf.keras.Input(shape=(4,))
    dx = tf.keras.layers.Dense(32, activation="relu")(latent)
    dx = tf.keras.layers.Dense(64, activation="relu")(dx)
    out = tf.keras.layers.Dense(input_dim, activation="linear")(dx)
    decoder = tf.keras.Model(latent, out, name="decoder")

    recon = decoder(z)
    loss_layer = VAELossLayer()([inp, recon, z_mean, z_logvar])
    vae = tf.keras.Model(inp, loss_layer, name="vae")
    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    return vae


def load_vae(path: str, dim: int) -> tf.keras.Model | None:
    if os.path.exists(path):
        try:
            m = tf.keras.models.load_model(path, custom_objects={
                "sampling": sampling, "VAELossLayer": VAELossLayer
            })
            if m.input_shape[1] != dim:
                raise ValueError(f"VAE expects {m.input_shape[1]} dims, got {dim}")
            return m
        except Exception:
            return None
    return None


def to_text(row: pd.Series) -> str:
    mem = row['MEM'] * 100
    return (
        f"Time {int(row['ts'])}: PID {int(row['PID'])}, "
        f"{int(row['MINFLT'])} minor faults, {int(row['MAJFLT'])} major faults, "
        f"{mem:.1f}% memory."
    )


def run_vae_single_shot(
    df: pd.DataFrame,
    progress_callback: Callable[[str], None] = print
) -> pd.DataFrame:
    """
    VAE-based anomaly detection with zero-shot classification (single-shot style).
    Progress is sent to progress_callback.

    Returns DataFrame with original columns plus:
      - anomaly_score
      - predicted_label
      - threshold
    """
    def log(msg: str):
        progress_callback(msg)

    log("[1/6] Preprocessing data...")
    features = ['ts','PID','MINFLT','MAJFLT','VSTEXT','VSIZE','RSIZE','VGROW','RGROW','MEM']
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required features: {missing}")

    df_clean = df.dropna(subset=features).reset_index(drop=True)
    X = df_clean[features].astype(float).values
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X).astype(np.float32)

    log("[2/6] Determining normal samples...")
    if 'type' in df_clean.columns:
        y = df_clean['type'].astype(int).values
        normal_mask = (y == 0)
    else:
        normal_mask = np.ones(len(X_scaled), dtype=bool)

    log("[3/6] Loading or training VAE...")
    model_path = "vae_single_shot_model.keras"
    vae = load_vae(model_path, X_scaled.shape[1])
    if vae is None:
        log("→ Training new VAE on normal data...")
        vae = build_vae(X_scaled.shape[1])
        vae.fit(
            X_scaled[normal_mask], X_scaled[normal_mask],
            epochs=20, batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        vae.save(model_path)
        log("✅ VAE training complete and saved.")

    log("[4/6] Computing reconstruction errors & threshold...")
    recon = vae.predict(X_scaled)
    errs = np.mean((X_scaled - recon)**2, axis=1)
    threshold = np.percentile(errs[normal_mask], 95)
    flags = errs > threshold
    log(f"→ Threshold set at {threshold:.4f}, flagged {flags.sum()} anomalies.")

    log("[5/6] Initializing zero-shot classifier...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )

    log("[6/6] Classifying flagged anomalies...")
    total = flags.sum()
    labeled = 0
    predictions = []
    for idx, flag in enumerate(flags):
        if flag:
            text = to_text(df_clean.iloc[idx])
            res = classifier(
                text,
                candidate_labels=["Normal","Anomaly"],
                hypothesis_template="This record is {}.",
                multi_label=False
            )
            label = res['labels'][0]
            labeled += 1
            if labeled % 10 == 0 or labeled == total:
                log(f"→ Classified {labeled}/{total}")
        else:
            label = "Normal"
        predictions.append(label)

    log("Building result DataFrame...")
    result = df_clean.copy()
    result['anomaly_score'] = errs
    result['predicted_label'] = predictions
    result['threshold'] = threshold
    log("✅ run_vae_single_shot finished.")
    return result
