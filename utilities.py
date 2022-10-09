import re
import random
import pandas as pd
import torch
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

#function to get the dictionary with probability of each words in the words list
def get_prob_dict(total_dataset, wordlist):
    #calculate the number of occurence for each words in the dataset
    counts = Counter() #counter object to count the no of words
    total_occurence = 0 #total occurence of words in the word list
    prob_dict = {} #dictionary to store the prob of each words in the words list i.e. P (h')
    #count the occurence of each words in the dataset
    for rows in total_dataset:
        counts.update(word.strip('.,?!"\'').lower() for word in rows.split())
    #calcualte the total occurence of all the words in the words list
    for word in wordlist:
        total_occurence = total_occurence + counts[word]
    #calculate the probability of each word occuring  p(h')
    for word in wordlist:
        prob = counts[word]/total_occurence
        prob_dict[word] = round(prob, 6)
    return prob_dict


#augment the given tweet by replacing the words present in the list with all the other words
def get_augmented_tweets_with_prob(tweet, wordlist, new_hate, prob_dict, label):
    st_singleword = []  
    single_tweets = []
    for words in wordlist:
        single_tweets = []
        ##randomly select n words to augment the tweets
        #new_hate = random.sample(wordlist, n_augment)
        #print(new_hate, '\n')
        if words in tweet.split():
            for check_words in new_hate:
                mod_tweet = re.sub(r"\b%s\b"%words, check_words, tweet)
                tweet_prob = (mod_tweet, prob_dict[check_words], label)
                single_tweets.append(tweet_prob)
            st_singleword.append(single_tweets)
            break
    return st_singleword

def misspell_hw(word):
    vowels = ['a', 'e', 'i', 'o', 'u']
    if len(word) < 4:
        lst = [char for char in word[1:]]
        random.shuffle(lst)
        lst.insert(0,word[0])
        misspell = ''.join(lst)
        return misspell
    
    elif len(word) >= 6:
        lst = [char for char in word[1:-1]]
        for vls in vowels:
            if vls in lst:
                new_vwls = [x for x in vowels if x != vls]
                mid_words = ''.join(lst)
                rand_vowel = random.sample(new_vwls, 1)
                mid_words = mid_words.replace(vls, rand_vowel[0], 1)
                lst = list(mid_words)
                lst.append(word[-1])
                lst.insert(0,word[0])
                misspell = ''.join(lst)
                return misspell
    else:
        lst = [char for char in word[1:-1]] 
        random.shuffle(lst)
        if lst[0] == lst[1] and len(lst) == 2:
            lst.append(word[-1])
            lst.insert(0,word[0])
            misspell = ''.join(lst)
            return misspell      
        else:
            lst.append(word[-1])
            lst.insert(0,word[0])
            misspell = ''.join(lst)
            if word == misspell or misspell == None:
                msp = misspell_hw(word)
                return msp
            else:
                return misspell


def get_misspell_dict(wordlist):
    #empty list to store the misspelling for each hate_words
    ms_hatewords = [] 
    #create misspelling for the hatewords
    for words in wordlist:
        misspell = misspell_hw(words)
        ms_hatewords.append(misspell)
    
    #key value dict for original and misspelled hate words
    res = {wordlist[i]: ms_hatewords[i] for i in range(len(wordlist))}
    return res

#returns the modified test by replacing hatewords in the tweets with one of the random words in the wordlist
def replace_test_hw(test_df, wordlist):
    mod_test = []
    for tweets in test_df['Sentence']:
        #print(tweets)
        flag = False
        for words in tweets.split():
            if words in wordlist:
                new_list = [x for x in wordlist if x!= words]
                random_word = random.sample(new_list, 1)
                mod_tweet = re.sub(r"\b%s\b"%words, random_word[0], tweets)
                mod_test.append(mod_tweet)
                flag = True
                break
        if flag == False:
            mod_test.append(tweets)
    mod_test_df = pd.DataFrame(zip(mod_test, test_df['Label']), columns = ['Sentence', 'Label'])
    return mod_test_df
            
#returns the modified test by misspelling hatewords in the tweets
def misspell_test_hw(test_df, wordlist, misspell_dic):
    misspell_test = []
    for tweets in test_df['Sentence']:
        flag = False
        for words in tweets.split():
            if words in wordlist:
                mod_tweet = re.sub(r"\b%s\b"%words, misspell_dic[words], tweets)
                misspell_test.append(mod_tweet)
                flag = True
                break
        if flag == False:
            misspell_test.append(tweets)
    misspell_test_df = pd.DataFrame(zip(misspell_test, test_df['Label']), columns = ['Sentence', 'Label'])
    return misspell_test_df

def load_model_from_checkpoints(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss

# ========================================================
#                  Preparing the data
# ========================================================

#tokenize the dataset and return the tokenized inputs
def tokenize_datasets(dataset, tokenizer):
    tokenized_inputs = []
    # For every tweet in the training set
    for tweet in dataset:
        encoded_tweet = tokenizer.encode(
            tweet,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=60,  # maximum length of the sequence
        )
        # Add the encoded tweet to the list.
        tokenized_inputs.append(encoded_tweet)
    return tokenized_inputs

#padding the tokenized inputs to a fix length
def padding_sequence(input_ids, tokenizer):
    return pad_sequences(input_ids, maxlen=60, dtype="long",
                          value=tokenizer.pad_token_id, truncating="pre", padding="pre")

#craete attention mask to distinguis between the padding and original ids
def create_attention_mask(input_ids):
    attention_mask = []
    # For each tweet in the training set
    for tweet in input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0. If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in tweet]
        # Store the attention mask for this sentence.
        attention_mask.append(att_mask)
    return attention_mask

#create dataloaders for training and testing
def get_dataloaders(train_df, test_df, tokenizer, args, wordlist):
    train_labels = train_df['Label'].tolist() #train dataset already encoded
    test_labels = test_df['Label'].tolist()  #small test dataset already encoded

    #test dataset with replaced words
    replaced_test_df = replace_test_hw(test_df, wordlist)

    #test dataset with misspelled words in the words list
    misspell_dict = get_misspell_dict(wordlist) #get the dictionary mapping misspelled words and original words
    misspelled_test_df = misspell_test_hw(test_df, wordlist, misspell_dict)
    #ms_words = list(misspell_dict.values()) #list of misspelled words only

    # List of all tokenized tweets
    train_input_ids = tokenize_datasets(train_df['Sentence'].tolist(), tokenizer)
    test_input_ids = tokenize_datasets(test_df['Sentence'].tolist(), tokenizer)
    misspelled_input_ids = tokenize_datasets(misspelled_test_df['Sentence'].tolist(), tokenizer)
    replaced_input_ids = tokenize_datasets(replaced_test_df['Sentence'].tolist(), tokenizer)

    #padding the input ids to make a fixed length of sequence
    train_input_ids = padding_sequence(train_input_ids, tokenizer)
    test_input_ids = padding_sequence(test_input_ids, tokenizer)
    misspelled_input_ids = padding_sequence(misspelled_input_ids, tokenizer)
    replaced_input_ids = padding_sequence(replaced_input_ids, tokenizer)

    # The attention mask simply makes it explicit which tokens are actual words versus which are padding
    train_attention_masks = create_attention_mask(train_input_ids)
    test_attention_masks = create_attention_mask(test_input_ids)
    misspelled_attention_masks = create_attention_mask(misspelled_input_ids)
    repalced_attention_masks = create_attention_mask(replaced_input_ids)

    #tweets id
    train_inputs = torch.tensor(train_input_ids)
    test_inputs = torch.tensor(test_input_ids)
    mis_inputs = torch.tensor(misspelled_input_ids)
    rep_inputs = torch.tensor(replaced_input_ids)

    #attention masks
    train_masks = torch.tensor(train_attention_masks)
    test_masks = torch.tensor(test_attention_masks)
    mis_masks = torch.tensor(misspelled_attention_masks)
    rep_masks = torch.tensor(repalced_attention_masks)

    #labels
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = SequentialSampler(train_data)
    train_dl = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    #create the dataloader for our original test dataset
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dl = DataLoader(test_data, sampler=test_sampler,  batch_size=args.batch_size)

    mis_data = TensorDataset(mis_inputs, mis_masks, test_labels)
    mis_sampler = SequentialSampler(mis_data)
    mis_dl = DataLoader(mis_data, sampler=mis_sampler,   batch_size=args.batch_size)

    rep_data = TensorDataset(rep_inputs, rep_masks, test_labels)
    rep_sampler = SequentialSampler(rep_data)
    rep_dl = DataLoader(rep_data, sampler=rep_sampler,   batch_size=args.batch_size)

    # Return the dataloaders
    return train_dl, test_dl, mis_dl, rep_dl