# Progetto NLP - Classificazione del Testo

## Autore
Raffaele Calì 

## Descrizione
Questo progetto esplora varie strategie per la classificazione del testo, categorizzando i testi in cinque categorie: business, sport, politica, tecnologia e scienza, e 'altro'. Il progetto combina tecniche di deep learning e machine learning classico per ottenere i migliori risultati possibili.

## Indice
1. [Introduzione](#introduzione)
2. [Dataset](#dataset)
3. [Tecniche Utilizzate e Risultati](#tecniche-utilizzate-e-risultati)
4. [Conclusioni e Sviluppi Futuri](#conclusioni-e-sviluppi-futuri)
5. [Demo](#demo)

## Introduzione
Il progetto si concentra sulla classificazione del testo, definita come il processo di categorizzazione del testo in gruppi organizzati. L'obiettivo è classificare i testi in cinque categorie utilizzando tecniche di deep learning, machine learning classico e una combinazione di entrambe.

## Dataset
Il dataset finale è una combinazione di tre dataset: BBC News, AG News e 20Newsgroup. Questi sono stati mappati in cinque categorie predefinite per garantire coerenza. Il dataset contiene circa 130.000 record con una distribuzione bilanciata tra le categorie.

### Struttura del Dataset
- **BBC News**: Articoli giornalistici formali.
- **AG News**: Sintesi di notizie brevi e incisive.
- **20Newsgroup**: Messaggi di forum con stile informale.

Il dataset è suddiviso in due parti:
- `dataset_k_neigh.csv`
- `dataset_Longformer.csv`

Un dataset secondario, `generated_pairs.csv`, è stato creato per il task di classificazione binaria.

## Tecniche Utilizzate e Risultati

### 1. Baseline: Estrazione Token CLS da Longformer e Classificazione con KNN
- **Procedura**: Preprocessing dei testi, estrazione degli embeddings CLS con Longformer, classificazione con KNN.
- **Risultati**:
  - Accuracy: 0.891243
  - Precision: 0.898594
  - Recall: 0.823507
  - F1: 0.850655

### 2. LDA e KNN
- **Procedura**: Configurazione dei topic con LDA, analisi dei testi, classificazione con KNN.
- **Risultati**:
  - Accuracy: 0.730223
  - Precision: 0.659870
  - Recall: 0.620796
  - F1: 0.625178

### 3. Fine-tuning di Longformer + KNN
- **Procedura**: Fine-tuning di Longformer su un task di classificazione binaria, estrazione degli embeddings CLS, classificazione con KNN.
- **Risultati**:
  - CLS: 
    - Accuracy: 0.891243
    - Precision: 0.898594
    - Recall: 0.823507
    - F1: 0.850655
  - CLS+LDA:
    - Accuracy: 0.882041
    - Precision: 0.879741
    - Recall: 0.820107
    - F1: 0.843028

### 4. Fine-tuning di Longformer con Classificazione Multiclasse
- **Procedura**: Fine-tuning di Longformer per la classificazione multiclasse.
- **Risultati**:
  - Accuracy: 0.8855
  - Precision: 0.8733
  - Recall: 0.8545
  - F1: 0.8624

## Conclusioni e Sviluppi Futuri
Il fine-tuning di Longformer per un compito di classificazione multiclasse ha mostrato le migliori prestazioni. Per sviluppi futuri, si consiglia di esplorare ulteriormente i metodi per gestire l'attenzione globale nei modelli transformers.

## Demo
Per provare la demo:
1. Esegui il file `demo.py`.
2. Inserisci il testo nella textarea.
3. Clicca su "Classifica testo" per vedere i risultati.

## Requisiti
- Python 3.x
- Librerie necessarie (vedi `requirements.txt`)

## Istruzioni di Installazione
1. Clona il repository:
   ```bash
   git clone https://github.com/tuo-username/tuo-repository.git
