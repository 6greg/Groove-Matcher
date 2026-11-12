# Groove Matcher

Dato un arrangiamento musicale (audio), **Groove Matcher** suggerisce le batterie MIDI della libreria che si abbinano meglio, cioè quelle con la struttura ritmica più correlata a quella dell'arrangiamento.

Il progetto nasce come lavoro di tesi. Il modello è una **Canonical Correlation Analysis (CCA)** a una componente, addestrata su coppie arrangiamento/batteria e valutata con test di permutazione.

## Come funziona

1. **Griglia temporale.** Si considerano 8 battute in 4/4 a risoluzione di sedicesimi: 128 step.
2. **Feature dell'arrangiamento** (5 canali per step, calcolati da STFT con `n_fft=2048`, `hop=512`, 44.1 kHz mono):
   - spectral flux in tre bande: basso (40–250 Hz), medio (250–4000 Hz), alto (4–20 kHz)
   - RMS
   - variazione del chroma (cambi armonici)
3. **Feature della batteria** (5 canali per step, dal file MIDI): `kick`, `snare`, `hihat`, `openhat`, `perc`. Il valore è la velocity normalizzata.
4. **CCA.** Per ogni tipo di arrangiamento un modello CCA proietta arrangiamento e batteria in una variabile canonica. Prima di questo, ogni canale viene centrato sulla sua media nel tempo.
5. **Matching.** Si calcola la correlazione di Pearson tra la variabile canonica dell'arrangiamento e quella di ciascuna delle 70 batterie della libreria. Vengono restituite le prime 5 (`top_k` in `plugin/groove_matcher.py`).

## Struttura del repository

```
plugin/      app Gradio (gui.py) e logica di matching (groove_matcher.py)
source/      cca.py: preprocessing, fit della CCA, correlazione canonica
notebooks/   pipeline di ricerca (vedi sotto)
data/
  train/     70 batterie MIDI + arrangiamenti audio di addestramento (non versionati)
  test/      17 batterie MIDI + arrangiamenti audio di test (non versionati)
  model/     modelli CCA addestrati (.joblib), uno per tipo di arrangiamento
  results/   tensori (.npz), risultati dei test (.parquet), grafici (.pdf)
```

I file audio `.wav` (circa 790 MB) e i MIDI delle batterie (`.mid`) **non sono nel repository**: restano solo in locale. Sono invece inclusi i tensori già estratti (`data/results/tensors/`) e i modelli già addestrati (`data/model/`).

L'app ha bisogno dei MIDI in `data/train/drums/` per restituire i file delle batterie suggerite, quindi vanno copiati a mano in quella cartella dopo aver clonato il repository.

## Tipi di arrangiamento

`bass`, `harm`, `full`, `full_no_bass`, `lead`. Nell'app sono disponibili `bass`, `harm`, `full` e `full_no_bass`. Il modello `lead` non è esposto perché non ha una correlazione significativa (vedi Risultati).

I file audio seguono la convenzione `sample_<mood>_<id>_<bpm>.wav`. I MIDI seguono `midi_<id>_<bpm>.mid`. L'`id` abbina arrangiamento e batteria.

## Installazione

Sviluppato con Python 3.14 (le versioni precedenti non sono state verificate).

```bash
pip install numpy scipy pandas scikit-learn joblib librosa pretty_midi gradio statsmodels pyarrow
```

## Uso dell'app

```bash
cd plugin
python gui.py
```

Nell'interfaccia:

1. carica un file audio dell'arrangiamento (lunga almeno 8 battute al BPM indicato)
2. inserisci il BPM (70–200)
3. scegli il tipo di arrangiamento

Ottieni una tabella con le 5 batterie più correlate, con il valore di correlazione, e i relativi file MIDI da scaricare.


### Da codice

```python
from groove_matcher import groove_matcher

tabella, midi = groove_matcher("arrangiamento.wav", bpm=120, arrangement_type="full")
```

## Riprodurre la pipeline

I notebook si eseguono da `notebooks/` e vanno lanciati in quest'ordine:

| Notebook | Cosa fa |
|---|---|
| `extract_vectors.ipynb` | estrae le feature da audio e MIDI e salva i tensori in `data/results/tensors/` (richiede gli `.wav` e i `.mid`) |
| `train_cca.ipynb` | addestra un modello CCA per tipo di arrangiamento e li salva in `data/model/` |
| `permutation_test.ipynb` | test di permutazione sul train (5000 permutazioni, seed 6) |
| `test_cca.ipynb` | test di permutazione sul set di test con i modelli già addestrati |
| `plots.ipynb` | grafici dei risultati in `data/results/plots/` |

Le correzioni per confronti multipli usano il metodo di Holm.

## Risultati

Correlazione canonica osservata contro la distribuzione nulla (5000 permutazioni delle batterie).

**Set di test** (modelli addestrati sul train):

| Tipo | n train | n test | Correlazione | Media nulla | p | p (Holm) | Significativo |
|---|---|---|---|---|---|---|---|
| bass | 32 | 8 | 0.413 | 0.277 | 0.0050 | 0.0102 | sì |
| full | 43 | 10 | 0.448 | 0.338 | 0.0018 | 0.0072 | sì |
| full_no_bass | 54 | 15 | 0.417 | 0.350 | 0.0014 | 0.0070 | sì |
| harm | 51 | 14 | 0.416 | 0.335 | 0.0034 | 0.0102 | sì |
| lead | 44 | 13 | 0.162 | 0.167 | 0.5861 | 0.5861 | no |

Per `bass`, `harm`, `full` e `full_no_bass` la relazione tra arrangiamento e batteria è significativa e si conferma sul set di test. Per `lead` non emerge alcuna relazione.

## Limiti

- Il campione è piccolo (70 batterie di training e 17 di test).
- La libreria di suggerimenti è limitata alle 70 batterie del training.
- Il modello descrive una correlazione strutturale, non un giudizio estetico.
