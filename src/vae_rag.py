# src/vae_rag.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import Callable

# --- VAE custom layers & builder ---
@tf.keras.utils.register_keras_serializable()
def sampling(args):
    mean, log_var = args
    log_var = tf.clip_by_value(log_var, -5.0, 5.0)
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
    inp = tf.keras.Input((input_dim,))
    x = tf.keras.layers.Dense(64, activation="relu")(inp)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    z_mean   = tf.keras.layers.Dense(4)(x)
    z_logvar = tf.keras.layers.Dense(4)(x)
    z        = tf.keras.layers.Lambda(sampling)([z_mean, z_logvar])

    latent = tf.keras.Input((4,))
    y = tf.keras.layers.Dense(32, activation="relu")(latent)
    y = tf.keras.layers.Dense(64, activation="relu")(y)
    out = tf.keras.layers.Dense(input_dim, activation="linear")(y)
    decoder = tf.keras.Model(latent, out)

    recon = decoder(z)
    loss_out = VAELossLayer()([inp, recon, z_mean, z_logvar])
    vae = tf.keras.Model(inp, loss_out)
    vae.compile(optimizer='adam')
    return vae

def load_vae(path: str, dim: int) -> tf.keras.Model | None:
    if os.path.exists(path):
        try:
            m = tf.keras.models.load_model(path,
                custom_objects={"sampling":sampling, "VAELossLayer":VAELossLayer})
            if m.input_shape[1] != dim:
                return None
            return m
        except Exception:
            return None
    return None

def to_text(row: pd.Series, features: list) -> str:
    d = {f: row[f] for f in features}
    return (
        f"Time {int(d['ts'])}: PID {int(d['PID'])}, "
        f"{int(d['MINFLT'])} minor faults, {int(d['MAJFLT'])} major faults, "
        f"{d['MEM']*100:.1f}% memory."
    )

def run_vae_rag(
    df: pd.DataFrame,
    progress_callback: Callable[[str], None] = print
) -> pd.DataFrame:
    def log(msg: str):
        progress_callback(msg)

    features = ['ts','PID','MINFLT','MAJFLT','VSTEXT','VSIZE','RSIZE','VGROW','RGROW','MEM']
    log("[1/7] Checking & preprocessing data…")
    missing = [c for c in features + ['type'] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    df_clean = df.dropna(subset=features+['type']).reset_index(drop=True)
    X = df_clean[features].astype(float).values
    y = df_clean['type'].astype(int).values

    log("[2/7] Scaling features…")
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)

    log("[3/7] Load/train VAE…")
    norm_mask = (y==0)
    model_path = "vae_rag_model.keras"
    vae = load_vae(model_path, Xs.shape[1])
    if vae is None:
        log("→ Training VAE on normals…")
        vae = build_vae(Xs.shape[1])
        vae.fit(Xs[norm_mask], Xs[norm_mask], epochs=20,
                batch_size=32, validation_split=0.1, verbose=0)
        vae.save(model_path)
        log("✅ VAE trained & saved.")

    log("[4/7] Computing reconstruction errors & threshold…")
    recon = vae.predict(Xs)
    errs = np.mean((Xs - recon)**2, axis=1)
    threshold = np.percentile(errs[norm_mask], 95)
    flags = errs > threshold
    log(f"→ Threshold={threshold:.4f}, {flags.sum()} flagged.")

    log("[5/7] Embedding records…")
    texts = [to_text(df_clean.iloc[i], features) for i in range(len(df_clean))]
    emb_model = SentenceTransformer('all-MiniLM-L6-v2')
    embs = emb_model.encode(texts, show_progress_bar=False).astype('float32')

    log("[6/7] Building nearest-neighbor indices…")
    k = 5
    norm_idx = np.where(y==0)[0]
    anom_idx = np.where(y==1)[0]
    nn_norm = NearestNeighbors(n_neighbors=k).fit(embs[norm_idx])
    nn_anom = NearestNeighbors(n_neighbors=k).fit(embs[anom_idx])

    log("[7/7] Classifying with RAG LLM…")
    rag = pipeline("text-generation",
                   model="EleutherAI/gpt-neo-2.7B",
                   tokenizer="EleutherAI/gpt-neo-2.7B",
                   device=-1)  # force CPU
    pred = ["Normal"] * len(df_clean)
    rationale = [""] * len(df_clean)
    total = flags.sum()
    done = 0
    for i in np.where(flags)[0]:
        # retrieve neighbors
        q = embs[i:i+1]
        nn = nn_norm.kneighbors(q, return_distance=False)[0]
        na = nn_anom.kneighbors(q, return_distance=False)[0]
        normals = [texts[norm_idx[j]] for j in nn]
        anomalies = [texts[anom_idx[j]] for j in na]
        prompt = (
            "NORMAL logs:\n" + "\n".join(normals) +
            "\n\nANOMALY logs:\n" + "\n".join(anomalies) +
            f"\n\nSuspect:\n{texts[i]}\n\nClassify:"
        )
        out = rag(prompt, max_new_tokens=20, do_sample=False,
                  truncation=True,
                  pad_token_id=rag.tokenizer.eos_token_id)[0]["generated_text"].strip()
        label = "Anomaly" if out.lower().startswith("anomaly") else "Normal"
        pred[i] = label
        rationale[i] = out
        done += 1
        if done % 5 == 0 or done == total:
            log(f"→ Classified {done}/{total}")

    log("Assembling result DataFrame…")
    res = df_clean.copy()
    res["anomaly_score"]   = errs
    res["predicted_label"] = pred
    res["threshold"]       = threshold
    res["rationale"]       = rationale
    return res
