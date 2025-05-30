import argparse
import warnings
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.optim as optim #added
from bert import BERT, BERTDataset
from rnn import RNN, RNNDataset
from ngram import Ngram
from preprocess import preprocessing_function

warnings.filterwarnings("ignore")


def prepare_data():
    # do not modify
    df_train = pd.read_csv('./data/IMDB_train.csv')
    df_test = pd.read_csv('./data/IMDB_test.csv')
    return df_train, df_test


def get_argument():
    # do not modify
    opt = argparse.ArgumentParser()
    opt.add_argument("--model_type",
                     type=str,
                     choices=['ngram', 'RNN', 'BERT'],
                     required=True,
                     help="model type")
    opt.add_argument("--preprocess",
                     type=int,
                     help="whether preprocessing, 0 means no and 1 means yes")
    opt.add_argument("--part",
                     type=int,
                     help="specify the part")

    config = vars(opt.parse_args())
    return config


def first_part(model_type, df_train, df_test, N):
    # load and train model
    if model_type == 'ngram':
        model = Ngram(N)
        model.train(df_train)
    else:
        raise NotImplementedError

    # test performance of model
    perplexity = model.compute_perplexity(df_test)
    print("Perplexity of {}: {}".format(model_type, perplexity))

    return model

def second_part(model_type, df_train, df_test, N):
    # training configure
    rnn_config = {
        'batch_size': 8,
        'epochs': 10,
        'lr': 5e-4,
        'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    }

    bert_config = {
        'batch_size': 4,
        'epochs': 1,
        'lr': 2e-5,
        'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    }
    

    
    # load and dataset and set model
    if model_type == 'ngram':
        model = first_part(model_type, df_train, df_test, N)

    elif model_type == 'RNN':
        # pad and truncate for this batch
        def collate_fn(data):
            sequences, labels = zip(*data)
            sequences, labels = list(sequences), list(labels)
            sequences = pad_sequence(sequences, batch_first=False, padding_value=0)
            sequences = sequences[:512, :]
            return sequences, torch.tensor(labels)
        
        train_data = RNNDataset(df_train)
        test_data = RNNDataset(df_test, train_data)
        train_dataloader = DataLoader(train_data, collate_fn=collate_fn, batch_size=rnn_config['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_data, collate_fn=collate_fn, batch_size=1, shuffle=False)
        rnn_config['vocab_size'] = len(train_data.vocab_word2int)
        model = RNN(config = rnn_config)

    elif model_type == 'BERT':
        model = BERT('distilbert-base-uncased', config=bert_config)

        def collate_fn(data):
            sequences, labels = zip(*data)
            sequences, labels = list(sequences), list(labels)
            sequences = model.tokenizer(sequences, padding = True, truncation = True, max_length = 512, return_tensors = "pt")
            return sequences, torch.tensor(labels)

        train_data = BERTDataset(df_train)
        test_data = BERTDataset(df_test)
        train_dataloader = DataLoader(train_data, batch_size = bert_config['batch_size'], collate_fn=collate_fn)
        test_dataloader = DataLoader(test_data, batch_size = 1, shuffle = False, collate_fn = collate_fn)

    else:
        raise NotImplementedError


    # train model
    if model_type == 'ngram':
        model.train_sentiment(df_train, df_test)
    elif model_type == 'BERT' or model_type == 'RNN':    
        train(
            model_type=model_type,
            model=model,
            train_dataloader=train_dataloader, 
            test_dataloader=test_dataloader,
            optimizer=torch.optim.Adam(model.parameters(), lr=model.config['lr']),
            loss_fn=nn.CrossEntropyLoss().to(model.config['device']),
            config=model.config
        )
    else:
        raise NotImplementedError

def train(model_type, model, train_dataloader, test_dataloader, optimizer, loss_fn, config):
    '''
    total_loss: the accumulated loss of training
    labels: the correct labels set for the test set
    pred: the predicted labels set for the test set
    '''
    # TO-DO 2-2: Implement the training function
    # BEGIN YOUR CODE
    cf_epoch = config['epochs']
    cf_device = config['device']
    print(f"Using {cf_device} device")


    criterion = nn.CrossEntropyLoss()#交叉熵主要是用來判定實際的輸出與期望的輸出接近程度
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(cf_epoch): # This is pseudo code
        model.train() #train mode
        train_loss = 0.0

        # training stage
        for batch in tqdm(train_dataloader): # This is pseudo code
            features, labels = batch #read in features and labels
            features, labels = features.to(cf_device), labels.to(cf_device) #move to gpu
            optimizer.zero_grad() #set the optimizer to zero grad every time
            output = model.forward(features) # get the output
            compute_loss = loss_fn(output, labels) #get the loss of the output

            compute_loss.backward() #backward propagate
            optimizer.step()
            #predicted = torch.max(output.data, 1)[1]

            train_loss += compute_loss.item() #add to total loss


        
        # testing stage
        model.eval() #evaluation mode
        with torch.no_grad():#no need to gradient when testing
            labels = []
            predicted_labels = []
            for batch in tqdm(test_dataloader): # This is pseudo code
                features, ans_labels = batch
                features, ans_labels = features.to(cf_device), ans_labels.to(cf_device) #move to gpu
                output = model.forward(features) # get the output

                pred_labels = output.argmax(dim=1).tolist()

                labels.extend(ans_labels.tolist())
                predicted_labels.extend(pred_labels)





        precision, recall, f1, support = precision_recall_fscore_support(labels, predicted_labels, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        avg_loss = round(train_loss / len(train_dataloader), 4)
        print(f"Epoch: {epoch}, F1 score: {f1}, Precision: {precision}, Recall: {recall}, Loss: {avg_loss}")
    
    # END YOUR CODE
    
if __name__ == '__main__':
    # get argument
    config = get_argument()
    model_type, preprocessed = config['model_type'], config['preprocess']
    N = 2  # we only use bi-gram in this assignment, but you can try different N

    # read and prepare data
    df_train, df_test = prepare_data()
    label_mapping = {'negative': 0, 'positive': 1}
    df_train['sentiment'] = df_train['sentiment'].map(label_mapping)
    df_test['sentiment'] = df_test['sentiment'].map(label_mapping)

    # feel free to add more text preprocessing method
    if preprocessed:
        df_train['review'] = df_train['review'].apply(preprocessing_function)
        df_test['review'] = df_test['review'].apply(preprocessing_function)

    if config['part'] == 1:
        # part 1: implement bi-gram model
        first_part(model_type, df_train, df_test, N)
    elif config['part'] == 2:
        # part 2: implement your nn model from scratch
        # part 3: implement your nn model from other pre-trained model
        second_part(model_type, df_train, df_test, N)