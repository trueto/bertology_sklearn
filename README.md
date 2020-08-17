# bertology_sklearn: a python toolkit for natural language understanding with BERT related models

## Key Features
- Easy Using
- Multiple NLP tasks
- BERTology models
- State of the art

## Easy Using
- Input Data
   - `X`
      - shape: `[n_samples, max_seq_len]` 
      - type: `list` | `ndarray` | `pd.Dataframe`
   - `y`
      - shape: `[n_samples]` | `[n_sampels, max_seq_len]` for NER | `[n_sampels, n_classes]` for multi-label classification
      - type: `list` | `ndarray`(numpy) | `Dataframe`(pandas)
      
- Classes
   - `BertologyClassifier`, used for customizing models for text classification
   - `BertologyTokenClassifier`, used for customizing models for natural language recognition(NER)

- Class methods
   - `fit(X,y)`, used for fine-tuning
   - `predict(X)`, used for predicting 
   - `score(X,y)`, used for scoring

- Easy training
   - early stopping
   - k-fold cross validation
   
## NLP tasks
- Name Entity Recognition
- Text Classification
   - Binary Classes
   - Multiple Classes 
   - Multi-Label Classes

## Bertology Models

## State of the art

## Using Examples

## Model Params

## Compare to other tools