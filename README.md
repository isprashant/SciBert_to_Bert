# Keyphrase Extraction using BERT 

Deep Keyphrase extraction using BERT.

## Usage

1. Clone this repository and install `pytorch-pretrained-BERT`.

~~2. From `scibert` repo, untar the weights (rename their weight dump file to `pytorch_model.bin`) and vocab file into a new folder `model`.~~

3. Change the parameters accordingly in `experiments/base_model/params.json`. We recommend keeping batch size of 4 and sequence length of 512, with 6 epochs, if GPU's VRAM is around 11 GB.
4. For training, run the command `python train.py --data_dir data/task1/ --model_dir experiments/base_model`
5. For eval, run the command, `python evaluate.py --data_dir data/task1/ --model_dir experiments/base_model --restore_file best`

## Results

### Subtask 1: Keyphrase Boundary Identification

We used IO format here. Unlike original SciBERT repo, we only use a simple linear layer on top of token embeddings.

On test set, we got:

1. **F1 score**: -
2. **Precision**: -
3. **Recall**: -
4. **Support**: -

### Subtask 2: Keyphrase Classification

We used BIO format here. Overall F1 score was 0.4981 on test set.

|          | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| Process  |     -     |   -    |    -     |    -    |
| Material |     -     |   -    |    -     |    -    |
| Task     |     -     |   -    |    -     |    -    |
| Avg      |     -     |   -    |    -     |    -    |

### Future Work

~~1. Some tokens have more than one annotations. We did not consider multi-label classification.~~<n>
~~2. We only considered a linear layer on top of BERT embeddings. We need to see whether SciBERT + BiLSTM + CRF makes a difference.~~
- [ ] 1. I want to convert this code as BERT and extract Keyphrases from text files.
- [ ] 2. The F1 score sould be achieved nearly as original work

## Credits

1. SciBERT: https://github.com/allenai/scibert
2. HuggingFace: https://github.com/huggingface/pytorch-pretrained-BERT
3. PyTorch NER: https://github.com/lemonhu/NER-BERT-pytorch
4. Pranav A: https://github.com/pranav-ust
4. BERT: https://github.com/google-research/bert
