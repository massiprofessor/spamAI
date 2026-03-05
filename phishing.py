"""
╔══════════════════════════════════════════════════════════════════╗
║  RILEVATORE PHISHING — APPRENDIMENTO CONTINUO                   ║
║  Dataset italiano + Enron (HuggingFace) + feedback utente       ║
╚══════════════════════════════════════════════════════════════════╝

Ogni email analizzata può essere confermata o corretta dall'utente.
Il feedback viene salvato in 'user_feedback.json' e il modello
viene riaddestrato automaticamente includendo i nuovi esempi.

PREREQUISITI:
    pip install scikit-learn pandas datasets tqdm

USO:
    python phishing_combined.py

COMANDI nel loop interattivo:
    esci  → termina la sessione
    stats → mostra statistiche aggiornate del modello
"""

import os
import sys
import json
import re
import time
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ─── PARAMETRI ────────────────────────────────────────────────
TRAIN_RATIO      = 0.70
RANDOM_SEED      = 42
MAX_ENRON_EMAILS = 33000
LOCAL_DATASET    = "emails_dataset.json"
FEEDBACK_FILE    = "user_feedback.json"
TFIDF_MAX_FEAT   = 60000
RETRAIN_EVERY    = 5        # riaddestra ogni N nuovi feedback
# ──────────────────────────────────────────────────────────────

# Colori disabilitati: compatibile con Windows CMD e PowerShell
G  = ""; R = ""; Y = ""
C  = ""; B = "";  RS = ""


# ══════════════════════════════════════════════════════════════
# GESTIONE FEEDBACK
# ══════════════════════════════════════════════════════════════
def load_feedback() -> list:
    if not os.path.exists(FEEDBACK_FILE):
        return []
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_feedback(feedback_list: list):
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback_list, f, indent=2, ensure_ascii=False)


def feedback_to_df(feedback_list: list) -> pd.DataFrame:
    if not feedback_list:
        return pd.DataFrame(columns=["text", "label", "source"])
    rows = [{
        "text":   fb["subject"] + " " + fb["body"],
        "label":  fb["label"],
        "source": "feedback"
    } for fb in feedback_list]
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════
def load_local() -> pd.DataFrame:
    if not os.path.exists(LOCAL_DATASET):
        print(f"{Y}[!] '{LOCAL_DATASET}' non trovato.{RS}")
        return pd.DataFrame(columns=["text", "label", "source"])
    with open(LOCAL_DATASET, "r", encoding="utf-8") as f:
        emails = json.load(f)
    rows = [{
        "text":   (e.get("subject", "") + " " + e.get("body", "")).strip(),
        "label":  e["label"],
        "source": "italiano"
    } for e in emails]
    df = pd.DataFrame(rows)
    print(f"  {G}✔{RS} Italiano: {len(df)} email  "
          f"(ham={G}{(df.label=='ham').sum()}{RS}  "
          f"spam={R}{(df.label=='spam').sum()}{RS})")
    return df


def load_enron_hf(max_emails: int) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        print(f"{R}[ERRORE]{RS} Installa con: pip install datasets tqdm")
        sys.exit(1)
    print(f"  {C}↓{RS} Download Enron da HuggingFace "
          f"(prima volta ~100 MB, poi in cache)...")
    ds = load_dataset("SetFit/enron_spam", split="train", trust_remote_code=True)
    rows = []
    for row in ds:
        label = "spam" if row.get("label", 0) == 1 else "ham"
        text  = ((row.get("subject", "") or "") + " " +
                 (row.get("text", "") or "")).strip()
        if text:
            rows.append({"text": text, "label": label, "source": "enron"})
        if len(rows) >= max_emails:
            break
    df = pd.DataFrame(rows)
    print(f"  {G}✔{RS} Enron: {len(df):,} email  "
          f"(ham={G}{(df.label=='ham').sum():,}{RS}  "
          f"spam={R}{(df.label=='spam').sum():,}{RS})")
    return df


def clean_text(text: str) -> str:
    text = re.sub(
        r"^(From|To|Cc|Bcc|Subject|Date|Message-ID|Content-Type|"
        r"MIME-Version|Return-Path|Received|X-[\w-]+)[^\n]*\n",
        "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"https?://\S+", " URL ",   text)
    text = re.sub(r"\S+@\S+",      " EMAIL ", text)
    text = re.sub(r"\d{5,}",       " NUM ",   text)
    text = re.sub(r"\s+",          " ",       text)
    return text.strip()[:3000]


# ══════════════════════════════════════════════════════════════
# ADDESTRAMENTO
# ══════════════════════════════════════════════════════════════
def build_model(df: pd.DataFrame, silent=False):
    df = df.copy()
    df["text"] = df["text"].fillna("").apply(clean_text)
    df = df[df["text"].str.len() > 10].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].values, df["label"].values,
        test_size=1 - TRAIN_RATIO,
        random_state=RANDOM_SEED,
        stratify=df["label"].values
    )

    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        min_df=2, max_features=TFIDF_MAX_FEAT,
        sublinear_tf=True, strip_accents="unicode"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    accuracy    = accuracy_score(y_test, predictions)
    spam_idx    = list(model.classes_).index("spam")
    cm          = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
    tn, fp, fn, tp = cm.ravel()

    stats = dict(
        total=len(df), accuracy=accuracy,
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
        train=len(X_train), test=len(X_test),
        n_feedback=int((df["source"] == "feedback").sum()),
        n_italiano=int((df["source"] == "italiano").sum()),
        n_enron=int((df["source"] == "enron").sum())
    )

    if not silent:
        _print_stats(df, stats, model, vectorizer)

    return model, vectorizer, spam_idx, accuracy, stats


def _print_stats(df, stats, model, vectorizer):
    print(f"\n{'='*65}")
    print(f"  {B}STATO DEL MODELLO{RS}")
    print(f"{'='*65}")
    print(f"  Totale email      : {B}{stats['total']:,}{RS}")
    print(f"  ✉️  Ham            : {G}{(df.label=='ham').sum():,}{RS}")
    print(f"  🚨 Spam           : {R}{(df.label=='spam').sum():,}{RS}")
    print(f"  📌 Italiano       : {stats['n_italiano']}")
    print(f"  📌 Enron          : {stats['n_enron']:,}")
    print(f"  👤 Feedback utente: {B}{stats['n_feedback']}{RS}")
    print(f"\n  🎯 Accuratezza    : {B}{stats['accuracy']*100:.2f}%{RS}")
    print(f"     Training: {stats['train']:,}  |  Test: {stats['test']:,}")
    print(f"\n  🔲 Matrice di confusione:")
    print(f"  {'':22} {'PREV. ham':>12} {'PREV. spam':>12}")
    print(f"  {'REALE ham':22} {stats['tn']:>12,} {stats['fp']:>12,}")
    print(f"  {'REALE spam':22} {stats['fn']:>12,} {stats['tp']:>12,}")
    print(f"\n  {G}TP spam rilevati     : {stats['tp']:,}{RS}")
    print(f"  {G}TN ham corretti      : {stats['tn']:,}{RS}")
    print(f"  {Y}FP falsi positivi    : {stats['fp']:,}  ← ham bloccate{RS}")
    print(f"  {R}FN spam non rilevati : {stats['fn']:,}  ← spam sfuggiti ⚠️{RS}")
    fn_arr   = vectorizer.get_feature_names_out()
    spam_lp  = model.feature_log_prob_[list(model.classes_).index("spam")]
    ham_lp   = model.feature_log_prob_[list(model.classes_).index("ham")]
    diff     = spam_lp - ham_lp
    top_idx  = diff.argsort()[-15:][::-1]
    print(f"\n  🔍 Top 15 parole {R}spam{RS} imparate dal modello:")
    for i, idx in enumerate(top_idx, 1):
        bar = f"{R}{'█' * min(int(diff[idx]), 18)}{RS}"
        print(f"  {i:>2}. {fn_arr[idx]:<28} {bar}")


# ══════════════════════════════════════════════════════════════
# LOOP INTERATTIVO CON FEEDBACK E RIADDESTRAMENTO
# ══════════════════════════════════════════════════════════════
def interactive_loop(df_base: pd.DataFrame, feedback_list: list):

    # Costruisce il modello iniziale (base + feedback pregressi)
    df_combined = pd.concat(
        [df_base, feedback_to_df(feedback_list)], ignore_index=True)
    model, vectorizer, spam_idx, accuracy, stats = build_model(df_combined)

    new_fb_count = 0   # feedback raccolti dall'ultimo riaddestramento

    print(f"\n{'='*65}")
    print(f"  {B}💡 ANALIZZATORE INTERATTIVO — Apprendimento Continuo{RS}")
    print(f"  Feedback salvati: {B}{len(feedback_list)}{RS}  |  "
          f"Accuratezza attuale: {B}{accuracy*100:.2f}%{RS}")
    print(f"  Il modello si riaddestra ogni {RETRAIN_EVERY} nuovi feedback.")
    print(f"  Comandi: '{B}esci{RS}' per uscire  |  '{B}stats{RS}' per statistiche")
    print(f"{'='*65}")

    while True:
        print()
        subject = input("📧 Oggetto: ").strip()

        # ── Comandi speciali ──────────────────────────────────
        if subject.lower() in ("esci", "exit", "quit", "q"):
            print(f"\n{G}👋 Sessione terminata.{RS}")
            print(f"   Feedback totali salvati : {B}{len(feedback_list)}{RS}")
            print(f"   File feedback           : {B}{FEEDBACK_FILE}{RS}")
            break

        if subject.lower() == "stats":
            df_now = pd.concat(
                [df_base, feedback_to_df(feedback_list)], ignore_index=True)
            build_model(df_now)
            continue

        if not subject:
            print(f"  {Y}⚠️  Oggetto vuoto.{RS}")
            continue

        # ── Input corpo ───────────────────────────────────────
        print("📝 Corpo (doppio INVIO per confermare):")
        body_lines = []
        while True:
            line = input()
            if line == "" and body_lines and body_lines[-1] == "":
                break
            body_lines.append(line)
        body = " ".join(body_lines).strip()

        if not body:
            print(f"  {Y}⚠️  Corpo vuoto.{RS}")
            continue

        # ── Classificazione ───────────────────────────────────
        text      = clean_text(subject + " " + body)
        vec       = vectorizer.transform([text])
        pred      = model.predict(vec)[0]
        proba     = model.predict_proba(vec)[0]
        conf_spam = proba[spam_idx]
        conf_ham  = 1 - conf_spam

        # ── Risultato ─────────────────────────────────────────
        print("\n" + "─" * 65)
        if pred == "spam":
            livello = ("ALTO"  if conf_spam >= 0.80 else
                       "MEDIO" if conf_spam >= 0.55 else "BASSO")
            nb    = int(conf_spam * 30)
            barra = f"{R}{'█'*nb}{'░'*(30-nb)}{RS}"
            print(f"  {R}{B}🚨 SPAM / PHISHING  —  Rischio {livello}{RS}")
            print(f"  Confidenza spam     : {R}{conf_spam*100:.1f}%{RS}")
            print(f"  Confidenza legittima: {conf_ham*100:.1f}%")
        else:
            nb    = int(conf_ham * 30)
            barra = f"{G}{'█'*nb}{'░'*(30-nb)}{RS}"
            print(f"  {G}{B}✅ EMAIL LEGITTIMA{RS}")
            print(f"  Confidenza legittima: {G}{conf_ham*100:.1f}%{RS}")
            print(f"  Confidenza spam     : {conf_spam*100:.1f}%")
        print(f"  [{barra}]")

        # ── Richiesta feedback ────────────────────────────────
        label_opposta = "legittima" if pred == "spam" else "spam"
        print(f"\n  {B}La previsione è corretta?{RS}")
        print(f"  [{B}S{RS}] Sì, confermo "
              f" [{B}N{RS}] No, è {label_opposta} "
              f" [{B}I{RS}] Ignora")

        while True:
            risposta = input("  → ").strip().lower()
            if risposta in ("s", "si", "sì", "y", "yes"):
                label_finale = pred
                salvare = True
                break
            elif risposta in ("n", "no"):
                label_finale = "ham" if pred == "spam" else "spam"
                salvare = True
                break
            elif risposta in ("i", "ignora", "skip", ""):
                salvare = False
                break
            else:
                print(f"  {Y}Digita S, N o I.{RS}")

        if not salvare:
            print(f"  {Y}↷ Ignorata, non salvata.{RS}")
            continue

        # ── Salvataggio ───────────────────────────────────────
        era_corretto = (label_finale == pred)
        entry = {
            "id":           len(feedback_list) + 1,
            "timestamp":    datetime.now().isoformat(),
            "subject":      subject,
            "body":         body,
            "label":        label_finale,
            "ai_predicted": pred,
            "was_correct":  era_corretto,
            "conf_spam":    round(conf_spam, 4),
            "conf_ham":     round(conf_ham, 4)
        }
        feedback_list.append(entry)
        save_feedback(feedback_list)
        new_fb_count += 1

        if era_corretto:
            print(f"  {G}✔ Confermato come {label_finale}. "
                  f"Feedback #{len(feedback_list)} salvato.{RS}")
        else:
            print(f"  {Y}✎ Corretto: {pred} → {label_finale}. "
                  f"Feedback #{len(feedback_list)} salvato.{RS}")

        # ── Riaddestramento automatico ────────────────────────
        if new_fb_count >= RETRAIN_EVERY:
            new_fb_count = 0
            print(f"\n  {C}🔄 Riaddestramento con {len(feedback_list)} "
                  f"feedback totali...{RS}", end="", flush=True)
            t0 = time.time()
            df_updated = pd.concat(
                [df_base, feedback_to_df(feedback_list)], ignore_index=True)
            model, vectorizer, spam_idx, accuracy, stats = build_model(
                df_updated, silent=True)
            print(f"\r  {G}✔ Modello aggiornato in {time.time()-t0:.1f}s  "
                  f"Accuratezza: {B}{accuracy*100:.2f}%{RS}  "
                  f"Dataset: {stats['total']:,} email  "
                  f"Feedback: {B}{stats['n_feedback']}{RS}")
        else:
            rimasti = RETRAIN_EVERY - new_fb_count
            print(f"  {C}ℹ  Ancora {rimasti} feedback al prossimo riaddestramento.{RS}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print(f"\n{'='*65}")
    print(f"  {B}RILEVATORE PHISHING — APPRENDIMENTO CONTINUO{RS}")
    print(f"  Italiano + Enron HuggingFace + feedback utente")
    print(f"{'='*65}\n")

    print(f"{C}[1/3]{RS} Dataset italiano...")
    df_local = load_local()

    print(f"\n{C}[2/3]{RS} Enron HuggingFace...")
    df_enron = load_enron_hf(MAX_ENRON_EMAILS)

    df_base = pd.concat([df_local, df_enron], ignore_index=True)

    print(f"\n{C}[3/3]{RS} Feedback utente salvati...")
    feedback_list = load_feedback()
    if feedback_list:
        corr  = sum(1 for f in feedback_list if f["was_correct"])
        wrong = len(feedback_list) - corr
        print(f"  {G}✔{RS} {len(feedback_list)} feedback trovati  "
              f"(confermati: {G}{corr}{RS}  corretti dall'utente: {Y}{wrong}{RS})")
    else:
        print(f"  {Y}Nessun feedback ancora — si parte da zero.{RS}")

    interactive_loop(df_base, feedback_list)


if __name__ == "__main__":
    main()
