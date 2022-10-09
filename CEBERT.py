import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import time
import datetime
from datetime import datetime as dt
import torch
from torch import nn
from utilities import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from utilities import *
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(5000)

# ==============================================================
#               CausalBERT Implementations
# ==============================================================
#detokenize the BERT input ids
def detokenize_bert_inputs(encoded_seq):
    tokens = ['[PAD]', '[CLS]', '[SEP]']
    decoded_seq = tokenizer.decode(encoded_seq) #detokenize the id and remove the first and last element
    wordlist = decoded_seq.split()
    seq_list = [x for x in wordlist if x not in tokens]
    return ' '.join(seq_list)
    
#calculate the L2 loss
def get_l2_loss(inputs, labels):
    b_out = torch.tensor(0.00)
    new_hate = random.sample(wordlist, args.n_augment)
    for twt_input, label in zip(inputs, labels):
        tweet = detokenize_bert_inputs(twt_input)
        augmented_tweets_with_prob = get_augmented_tweets_with_prob(tweet, wordlist, new_hate ,prob_dict, label)
        aug_out = get_augmented_tweets_output(augmented_tweets_with_prob, tweet)
        b_out = b_out + aug_out
    return b_out

#get the final ouputs i.e. Summation f(xj , h') * P (h')
def get_augmented_tweets_output(augmented_tweets_with_prob, tweet):
    
    #for each list of augmented tweets for words present in the tweets
    for i in range(len(augmented_tweets_with_prob)):        
        #index 0 contains the augmented tweets and 1 contains the probability for the words replaced
        augmented_tweets = [tweets[0] for tweets in augmented_tweets_with_prob[i]]
        prob = [tweets[1] for tweets in augmented_tweets_with_prob[i]]
        label = [tweets[2] for tweets in augmented_tweets_with_prob[i]]
        #normalize the probability
        probs = [pb/sum(prob) for pb in prob] 
        #tokenize the augmented tweets
        tokenized_input_ids = tokenize_datasets(augmented_tweets, tokenizer)
        #padding augmented tweets to the fized length
        tokenized_input_ids = padding_sequence(tokenized_input_ids, tokenizer)
        #get the attention masks
        tokenized_attention_masks = create_attention_mask(tokenized_input_ids)
        #convert the input ids, attention mask and label to tensors
        tokenized_inputs = torch.tensor(tokenized_input_ids)
        tokenized_masks = torch.tensor(tokenized_attention_masks)
        probs = torch.tensor(probs)
        labels = torch.tensor(label)
        #create the dataset and data loader
        augmented_data = TensorDataset(tokenized_inputs, tokenized_masks, probs, labels)
        augmented_sampler = SequentialSampler(augmented_data)
        augmented_loader = DataLoader(augmented_data, sampler = augmented_sampler,   batch_size= args.n_augment)
        total_loss = 0
        for step, batch in enumerate(augmented_loader):
            b_input_ids = batch[0].to(device) 
            b_input_mask = batch[1].to(device)
            b_probs = batch[2].to(device)
            b_labels = batch[3].to(device)
            model.zero_grad()
            bert_ouput = model(b_input_ids,
                            token_type_ids = None,
                            attention_mask = b_input_mask,
                            labels = b_labels)
            soft_out = softmax(bert_ouput[1])
            bert_probs = soft_out[:,1]
            #part_1 = torch.mul(torch.log(torch.sum(output)), torch.sum(probs.to(device)))
            #part_2 = torch.mul(torch.log(torch.sum(1-output)), torch.sum(probs.to(device)))
            part_1 = torch.log(torch.sum(torch.mul(bert_probs, b_probs.to(device))))
            part_2 = torch.log(torch.sum(torch.mul((1-bert_probs), b_probs.to(device))))
            loss = (-1/args.batch_size) * ((b_labels[0] * part_1)+ ((1 - b_labels[0]) * part_2))
            total_loss += loss
        return total_loss

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#Takes a time in seconds and returns a string hh:mm:ss
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# ======================================================
#                        Training
# ======================================================
def start_training():
    model.train()
    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    for epoch_i in range(0, args.epoch):
        print("\nEpoch: {}/{}...".format(epoch_i+1, args.epoch))
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_loss = 0

        for step, batch in enumerate(train_dl):
            #get the data from batch and move to GPUS
            b_input_ids = batch[0].to(device) 
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            #clear any previously calculated gradients 
            #model.zero_grad()
            
            if args.model_type == 'baseBERT':
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                loss = outputs[0]
            else:
                if args.lamda == 1:
                    loss = get_l2_loss(b_input_ids, b_labels)
                else:    
                    outputs = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                    l1_loss = outputs[0]
                    l2_loss = get_l2_loss(b_input_ids, b_labels)
                    loss = (1 - args.lamda) * l1_loss +  (args.lamda * l2_loss)

            # Progress update every 50 batches.
            if step % 150 == 0 and not step == 0:
                # Report progress.
                print(' Batch {:>5,}  of  {:>5,}: Loss = {:.5f}'.format(step, len(train_dl), loss))
            #add the loss of all the batch to get the average loss of an epoch
            total_loss += loss

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dl)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        
        print("    Average training loss: {0:.2f}".format(avg_train_loss))
        print("    Training epoch took: {:}".format(format_time(time.time() - t0)))


# ==============================================================
#               Evaluation
# ==============================================================
def test_model(dataloader):
    # Store true lables for global eval
    gold_labels = []
    # Store  predicted labels for global eval
    predicted_labels = []

    model.eval()
    # Tracking variables
    nb_eval_steps, eval_accuracy = 0, 0

    #evaluate data for one epoch
    for batch in dataloader:
        # Add batch to GPU/CPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients
        with torch.no_grad():
            #get the output of the BERT
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # logits are the values prior to applying an 
        # activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        
        # Store gold labels single list
        gold_labels.extend(labels_flat)
        # Store predicted labels single list
        predicted_labels.extend(pred_flat) 

    # Report the final accuracy for this validation run.
    cm = confusion_matrix(gold_labels, predicted_labels)
    print('Confusion Matrix: \n', cm)
    print('\nAccuracy: {:.3f}'.format(accuracy_score(gold_labels, predicted_labels)))
    print('Precision: {:.3f}'.format(precision_score(gold_labels, predicted_labels)))
    print('Recall: {:.3f}'.format(recall_score(gold_labels, predicted_labels)))
    print('F1 Score: {:.3f} \n'.format(f1_score(gold_labels, predicted_labels)))
    print(classification_report(gold_labels, predicted_labels))


# ==============================================================
#               Main Iteration of the program
# ==============================================================
#create a parser to parse tha arguments passed
parser = argparse.ArgumentParser()
#defining arguments
parser.add_argument("--n_augment",
                    type= int,
                    default = 16,
                    help="No of words to augment during training [Only for CEBERT]")

parser.add_argument('--lamda', 
                    type = float,
                    default = 0.5,
                    help = 'Regularization term to balance model utility and model robustness. range [0 - 1] [Only for CEBERT]')

parser.add_argument('--epoch',
                    type = int,
                    default = 4,
                    help = 'Number of iterations to train the model')

parser.add_argument('--batch_size',
                    type = int,
                    default = 32,
                    help = 'Size of the batch')

parser.add_argument('--model_type',
                    type = str,
                    default = 'baseBERT',
                    choices = ['baseBERT', 'CEBERT'],
                    help = 'Type of the model to train the data on.')

args = parser.parse_args()  
print('\n {} \n'.format(args))

#Compute in GPU if found else use CPU
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('\n Device being used: {} \n'.format(device))   
#load the list of hate and positive words
hw_pw_list = pd.read_csv('data/words.csv')
wordlist = hw_pw_list['Words'].tolist() #convert the series of words into list
#load the training and test dataset
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
#clean the dataset and get the probability for each words
clean_train = train_df['Sentence'].tolist()
prob_dict = get_prob_dict(clean_train, wordlist)

# ----------------------------------------------------------------------
# -------------- Loading optimizer and modelfor training ---------------
# ----------------------------------------------------------------------
config_class, model_class, tokenizer_class = (BertConfig, BertForSequenceClassification, BertTokenizer)
# Load pre-trained Tokenizer from directory, change this to load a tokenizer from ber package
tokenizer = tokenizer_class.from_pretrained("bert-base-uncased") 
#define a softmax function
softmax = nn.Softmax(dim = 1)
#data loaders
train_dl, test_dl, mis_dl, rep_dl = get_dataloaders(train_df, test_df, tokenizer, args, wordlist)
# Load Bert for classification 'container'
model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use pre-trained model from its directory, change this to use a pre-trained model from bert
        num_labels = 2, # The number of output labels--2 for binary classification.
        output_attentions = True, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
)
#put the model into GPU
model.to(device)
print("\n BERT Model Loaded !!! \n")

optimizer = torch.optim.AdamW(model.parameters(),
                lr=1e-5,  # args.learning_rate 
                eps=1e-8  # args.adam_epsilon  - default is 1e-8
                )

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dl) * args.epoch 
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=0,  # Default value in run_glue.py
                                        num_training_steps=total_steps)
   
test_loaders = (test_dl, rep_dl, mis_dl)
test_texts = ('\n\nEvaluating Original Test Dataset: \n',
              '\n\nEvaluating Original Replaced Dataset: \n',
              '\n\nEvaluating Original Misspelled Dataset: \n')

start_training()
for test_loader, message in zip(test_loaders, test_texts):
    print('\n\n******************************************************************')
    print(message)
    test_model(test_loader)
    print('*********************************************************************')

