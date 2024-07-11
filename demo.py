from flask import Flask, request, jsonify, render_template
import torch
from transformers import LongformerModel,LongformerTokenizerFast
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
import os
import numpy as np

import joblib
import gensim
from torch.nn import BCEWithLogitsLoss
from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel, LongformerClassificationHead

from transformers import PreTrainedModel
from torch.nn.functional import cosine_similarity
from torch import nn
from tqdm import tqdm

app = Flask(__name__)

class SingleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        if idx >= len(self.texts):
            raise IndexError("Index out of range")
        
        text = self.texts[idx]
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }

    def __len__(self):
        return len(self.texts)

class ModifiedModelForBinaryClassification(LongformerPreTrainedModel):
    def __init__(self, config):
        super(ModifiedModelForBinaryClassification, self).__init__(config)
        self.longformer = LongformerModel(config)
        self.init_weights()
    
    def forward(self, input_ids_1, attention_mask_1, input_ids_2=None, attention_mask_2=None, labels=None):
    # global attention mask con attenzione solo sul primo token 
        global_attention_mask_1 = torch.zeros_like(input_ids_1)
        global_attention_mask_1[:, 0] = 1 
        
        outputs_1 = self.longformer(input_ids_1, attention_mask=attention_mask_1, 
                                    global_attention_mask=global_attention_mask_1)
        sequence_output_1 = outputs_1['last_hidden_state']
        cls_token_1 = sequence_output_1[:, 0, :]  
        
        if input_ids_2 is None:
            return cls_token_1
        
        
        global_attention_mask_2 = torch.zeros_like(input_ids_2)
        global_attention_mask_2[:, 0] = 1  
        
        outputs_2 = self.longformer(input_ids_2, attention_mask=attention_mask_2, 
                                    global_attention_mask=global_attention_mask_2)
        sequence_output_2 = outputs_2['last_hidden_state']
        cls_token_2 = sequence_output_2[:, 0, :]  
        
        logits = cosine_similarity(cls_token_1, cls_token_2).unsqueeze(-1)  # Assicurati che i logits siano della forma [batch_size, 1]

        loss = None
        if labels is not None:
                loss_fct = nn.BCEWithLogitsLoss()
                labels = labels.view(-1, 1).float()  # Ridimensiona le etichette per batch_size =1
                loss = loss_fct(logits, labels)

        return (loss, logits) if loss is not None else logits


class Data_Processing(object):
    def __init__(self, tokenizer, dataframe):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        # Nomi delle colonne codificati direttamente nella classe
        self.text_column_name = 'Text'  # Sostituisci con il nome effettivo della colonna
        self.label_column_name = 'Category_OneHot'  # Sostituisci con il nome effettivo della colonna

    def __getitem__(self, idx):
        # Accede direttamente alla riga del DataFrame utilizzando l'indice
        row = self.dataframe.iloc[idx]

        comment_text = str(row[self.text_column_name])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            add_special_tokens=True,
            max_length=1024,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze()  # Rimuove l'eventuale dimensione extra
        attention_mask = inputs['attention_mask'].squeeze()
        labels_ = torch.tensor(row[self.label_column_name], dtype=torch.float)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels_,
            'id_': row.name  # `row.name` contiene l'indice della riga
        }

    def __len__(self):
        return len(self.dataframe)
    
class LongformerForMultiLabelSequenceClassification(LongformerPreTrainedModel):
  
    def __init__(self, config):
        super(LongformerForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.longformer = LongformerModel(config)
        self.classifier = LongformerClassificationHead(config)
        self.init_weights()
    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None,
                token_type_ids=None, position_ids=None, inputs_embeds=None,
                labels=None):

        
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids = input_ids,
            attention_mask = attention_mask,
            global_attention_mask = global_attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids)

        
        sequence_output = outputs['last_hidden_state']

        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))
            #outputs = (loss,) + outputs
            outputs = (loss,) + outputs

        return outputs

device = torch.device("cpu")

modelmulticlass = LongformerForMultiLabelSequenceClassification.from_pretrained(os.getcwd() +"/longmulticlass/best_model/kaggle/working/best_model")
modelbinary = ModifiedModelForBinaryClassification.from_pretrained(os.getcwd() + "/binarymodel/modelraffa/checkpoint-3750")

tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096',
                                                    padding='max_length',
                                                    truncation=True,
                                                    max_length=1024,
                                                   )


model_filename = 'knn_model.joblib'

knn_loaded = joblib.load(os.getcwd() +'/knn_model/modello_knn.joblib')
knn_loaded_lda = joblib.load(os.getcwd() + '/knn_model_lda/modello_knn_lda.joblib')

loaded_lda_model = gensim.models.LdaModel.load(os.getcwd()+"/knn_model_lda/lda_model")
loaded_dictionary = gensim.corpora.Dictionary.load(os.getcwd()+'/knn_model_lda/dictionary.dict')
loaded_corpus = gensim.corpora.MmCorpus(os.getcwd()+'/knn_model_lda/corpus.mm')

def get_lda_topic_scores(document, dictionary, lda_model):
    bow = dictionary.doc2bow(document.split())
    topic_scores = np.zeros(lda_model.num_topics)
    for topic, score in lda_model.get_document_topics(bow):
        topic_scores[topic] = score
    return topic_scores

def normalize_scores(scores):
    score_sum = sum(scores)
    if score_sum > 0:
        return [score / score_sum for score in scores]
    else:
        return [0 for score in scores]

def augment_embeddings_with_topics(embeddings, normalized_scores):
    embeddings = np.atleast_2d(embeddings)  
    normalized_scores = np.atleast_2d(normalized_scores)  
    
    if embeddings.shape[0] != normalized_scores.shape[0]:
        raise ValueError("Il numero di embeddings non corrisponde al numero di set di punteggi dei topic.")
    
    augmented_embeddings = np.concatenate([embeddings, normalized_scores], axis=1)
    
    return augmented_embeddings

def predict_with_multilabel(text: str, model: PreTrainedModel, inputs) -> int:
    model.eval()
    global_attention_mask = torch.zeros_like(inputs['input_ids'])
    global_attention_mask[:, 0] = 1
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    global_attention_mask = global_attention_mask.to(device)
    with torch.no_grad():
        outputs = model(**inputs, global_attention_mask=global_attention_mask)
        predictions = torch.nn.functional.softmax(outputs[0], dim=-1)
    predicted_class_id = np.argmax(predictions.cpu().numpy(), axis=1)[0]
    
    
    category_to_index = {'business': 0, 'other': 1, 'politics': 2, 'sport': 3, 'tech': 4}               
    index_to_category = {v: k for k, v in category_to_index.items()}
    predicted_category = index_to_category[predicted_class_id]
    
    
    
    
    return predicted_category

def predict_with_binary(text: str, model: PreTrainedModel, inputs) -> torch.Tensor:
    
    model.eval()
    #inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids_1=inputs['input_ids'], attention_mask_1=attention_mask)
        
    return outputs



os.environ["TOKENIZERS_PARALLELISM"] = "false"







@app.route('/')
def home():
     return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_text():
    # Estrai il testo dalla richiesta
    data = request.json
    text = data['text']
    print(text)
    if len(text) < 5:
        ris = "Fornire Testo Valido"
        return jsonify({'class': ris})
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    #multilabel    
    label_multilabel  = predict_with_multilabel(text,modelmulticlass,inputs)

    #knn solo emb
    cls_emb_to_knn = predict_with_binary(text,modelbinary,inputs)
    prediction = knn_loaded.predict(cls_emb_to_knn)
    # knn ldda
    lda_scores = get_lda_topic_scores(text, loaded_dictionary, loaded_lda_model)
    normalized_scores = normalize_scores(lda_scores)
    augmented_embeddings = augment_embeddings_with_topics(cls_emb_to_knn, normalized_scores)
    prediction_lda = knn_loaded_lda.predict(augmented_embeddings)
    ris = "Multilabel longformer: " + label_multilabel + " |-| binary cls emb knn: "+ prediction[0] + "  |-| knn + lda: " + prediction_lda[0]
    print(ris)
    return jsonify({'class': ris})
if __name__ == '__main__':
    app.run(debug=True)
