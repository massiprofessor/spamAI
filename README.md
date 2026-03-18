# 🛡️ spamAI — Rilevatore Phishing con Apprendimento Continuo

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Dataset-1000%20email%20italiane-green" alt="Dataset"/>
  <img src="https://img.shields.io/badge/Accuratezza-98%25%2B-brightgreen" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/Licenza-MIT-lightgrey" alt="License"/>
</p>

> Classificatore email spam/phishing in italiano basato su **Naive Bayes + TF-IDF**, addestrato su dataset italiano + Enron (~33.000 email reali) con sistema di **apprendimento continuo dal feedback dell'utente**.

---

## ✨ Caratteristiche principali

- 🇮🇹 **Dataset italiano** — 1.000 email in italiano (spam e ham) create appositamente per il contesto italiano (INPS, banche, operatori telefonici, truffe comuni)
- 🌍 **Integrazione Enron** — ~33.000 email reali scaricate automaticamente da HuggingFace (`SetFit/enron_spam`)
- 🤖 **Apprendimento continuo** — ogni email analizzata può essere confermata o corretta dall'utente; il modello si riaddestra automaticamente ogni 5 feedback
- 💾 **Feedback persistente** — i feedback vengono salvati in `user_feedback.json` e ricaricati ad ogni avvio
- 📊 **Statistiche in tempo reale** — accuratezza, matrice di confusione, top parole spam
- 🔄 **Riaddestramento automatico** — il modello migliora ad ogni sessione

---

## 🗂️ Struttura del progetto

```
spamAI/
├── phishing.py           # Script principale
├── emails_dataset.json   # 1.000 email italiane etichettate (spam/ham)
├── user_feedback.json    # Feedback utente accumulati (cresce nel tempo)
└── README.md
```

---

## 🚀 Avvio rapido

### 1. Clona il repository

```bash
git clone https://github.com/massiprofessor/spamAI.git
cd spamAI
```

### 2. Installa le dipendenze

```bash
pip install scikit-learn pandas datasets tqdm
```

### 3. Avvia il classificatore

```bash
python phishing.py
```

Al primo avvio scarica il dataset Enron da HuggingFace (~100 MB, poi in cache). Dalla seconda esecuzione parte istantaneamente.

---

## 🖥️ Utilizzo

All'avvio lo script mostra le statistiche del modello e poi entra in modalità interattiva:

```
📧 Oggetto: Hai vinto un iPhone! Ritira subito il premio
📝 Corpo (doppio INVIO per confermare):
Congratulazioni! Sei stato selezionato...
[INVIO]
[INVIO]

─────────────────────────────────────────────────────────────
  SPAM / PHISHING  —  Rischio ALTO
  Confidenza spam     : 97.3%
  Confidenza legittima: 2.7%
  [██████████████████████████████]

  La previsione è corretta?
  [S] Sì, confermo   [N] No, è legittima   [I] Ignora
  → S
  Confermato come spam. Feedback #1 salvato.
  Ancora 4 feedback al prossimo riaddestramento.
```

### Comandi disponibili

| Comando | Descrizione |
|---------|-------------|
| `esci` | Termina la sessione |
| `stats` | Mostra statistiche aggiornate del modello |
| `S` / `Y` | Conferma la previsione del modello |
| `N` | Correggi la previsione (insegna al modello) |
| `I` | Ignora — non salvare questo caso |

---

## 🧠 Come funziona

### Pipeline di classificazione

```
Email (oggetto + corpo)
        │
        ▼
   Pulizia testo
   (rimozione header, URL→URL, email→EMAIL, numeri→NUM)
        │
        ▼
   TF-IDF Vectorizer
   (60.000 feature, ngram 1-2, sublinear_tf)
        │
        ▼
   Naive Bayes (MultinomialNB, alpha=0.1)
        │
        ▼
   Probabilità spam/ham + barra visiva
```

### Dataset combinato

| Sorgente | Email | Lingua | Note |
|----------|-------|--------|------|
| `emails_dataset.json` | 1.000 | 🇮🇹 Italiano | Creato appositamente per il contesto italiano |
| `SetFit/enron_spam` | ~33.000 | 🇺🇸 Inglese | Email reali aziendali Enron |
| `user_feedback.json` | variabile | 🇮🇹 / 🌍 | Feedback utente — cresce ad ogni sessione |

### Apprendimento continuo

Ogni feedback confermato o corretto dall'utente viene:
1. Salvato in `user_feedback.json` con timestamp, predizione AI e etichetta corretta
2. Incluso nel dataset di training alla successiva sessione
3. Usato per riaddestramento automatico ogni 5 nuovi feedback

```json
{
  "id": 42,
  "timestamp": "2026-03-18T10:32:11",
  "subject": "Urgente: verifica il tuo conto UniCredit",
  "body": "Clicca qui per verificare...",
  "label": "spam",
  "ai_predicted": "spam",
  "was_correct": true,
  "conf_spam": 0.961,
  "conf_ham": 0.039
}
```

---

## 📈 Performance

| Metrica | Valore |
|---------|--------|
| Accuratezza (test set) | **98%+** |
| Dataset training | ~23.000 email |
| Dataset test | ~10.000 email |
| Split | 70% train / 30% test (stratificato) |
| Feature TF-IDF | 60.000 |

---

## ⚙️ Parametri configurabili

All'inizio di `phishing.py`:

```python
TRAIN_RATIO      = 0.70    # percentuale dati di training
RANDOM_SEED      = 42      # riproducibilità
MAX_ENRON_EMAILS = 33000   # max email Enron da caricare
TFIDF_MAX_FEAT   = 60000   # feature TF-IDF
RETRAIN_EVERY    = 5       # feedback prima del riaddestramento
```

---

## 🗃️ Formato emails_dataset.json

```json
[
  {
    "id": 1,
    "subject": "URGENTE: Il tuo conto è stato sospeso",
    "body": "Gentile cliente, abbiamo rilevato attività sospetta...",
    "label": "spam"
  },
  {
    "id": 2,
    "subject": "Riunione di team - giovedì ore 15",
    "body": "Ciao a tutti, vi convoco per una riunione...",
    "label": "ham"
  }
]
```

I valori validi per `label` sono `"spam"` e `"ham"`.

---

## 🔧 Requisiti

- Python 3.8+
- `scikit-learn`
- `pandas`
- `datasets` (HuggingFace)
- `tqdm`

```bash
pip install scikit-learn pandas datasets tqdm
```

---

## 📋 Casi d'uso

- **Studenti di cybersecurity** — laboratorio pratico su ML applicato al rilevamento phishing
- **Docenti** — materiale didattico pronto all'uso con dataset in italiano
- **Uso personale** — analizza le email sospette copiando oggetto e corpo

---

## 🤝 Contribuire

Contributi benvenuti! In particolare:

- Nuove email italiane per `emails_dataset.json` (spam o ham realistiche)
- Miglioramenti all'algoritmo di classificazione
- Supporto per altri dataset pubblici

---

## 📄 Licenza

MIT License — libero utilizzo, anche per scopi didattici.

---

<p align="center">
  Realizzato con 🐍 Python · scikit-learn · HuggingFace Datasets
</p>
