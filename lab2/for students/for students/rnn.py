import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from collections import Counter

class RNNDataset(Dataset):
    def __init__(self, df, train_data=None):
        super().__init__()

        self.reviews = df['review'].values
        self.sentiments = df['sentiment'].values

        # tokenize each sentence
        tokenzier = get_tokenizer("basic_english")
        for idx in range(len(self.reviews)):
            self.reviews[idx] = ' '.join(tokenzier(self.reviews[idx]))

        if train_data is None:
            # build vocab
            words = ' '.join(self.reviews)
            words = words.split()
            counter = Counter(words)
            vocab = sorted(counter, key=counter.get, reverse=True)

            self.vocab_int2word = dict(enumerate(vocab, 2))
            self.vocab_int2word[0] = '<PAD>'
            self.vocab_int2word[1] = '<UNK>'
            self.vocab_word2int = {word: id for id, word in self.vocab_int2word.items()}
        else:
            self.vocab_int2word = train_data.vocab_int2word
            self.vocab_word2int = train_data.vocab_word2int

        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review, sentiment = self.reviews[idx], self.sentiments[idx]
        review = [self.vocab_word2int.get(word, 1) for word in review.split()]
        return (torch.tensor(review), torch.tensor(sentiment))

class YourModel(nn.Module):
    def __init__(
            self,

            # TO-DO 3-1: Pass parameters to initialize model
            # BEGIN YOUR CODE
            embedding_dim, # dimension of embedding(embedding: translating from word to word vector) the vector consists of n numbers representing the vector
            #if = 500, (x0,x1,...,x499) to represent a word vector
            num_embeddings, #the size of the dictionary (if = 5000, there will be 5000 words)
            num_layers ,
            optimizer_type,
            #loss_function,
            #regularization,
            hidden_size,
            dropout,
            learning_rate, #learning rate設置過小時，會導致模型學習收斂較慢，且容易過擬合。而設置過大時，會容易導致模型學習過快，無法收斂，容易震盪甚至發散
            bidirectional = True
            # END YOUR CODE
        ):
        super().__init__()

        
        '''
        When we want to build a model using PyTorch, we first create a class and make it inherit from torch.nn.Module. 
        Then, there are two important methods that we need to define ourselves: init() and forward(). 
        In the init() function, we need to define which components our model consists of. 
        In forward(), we need to define how the input data to the model interacts with the components
        defined in init() to generate the final output.
        
        For example, in a sentiment classification model, we typically need at least:
        
        1. torch.nn.Embedding()
        2. torch.nn.RNN()
        3. torch.nn.Linear()
        
        Note that in this part: 

        1. Please use torch.nn to create the components required for your model. 
        
        2. please do not use any pre-trained models or components. In other words, the parameters of 
        the components you define must be randomly initialized rather than pre-trained by others.
        '''
        # TO-DO 3-2: Determine which modules your model should consist of
        # BEGIN YOUR CODE
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = nn.Embedding(embedding_dim=embedding_dim, num_embeddings= num_embeddings).to(device)
        self.rnn = nn.RNN(
            hidden_size = hidden_size,
            input_size = embedding_dim,
            num_layers = num_layers,
            dropout = dropout,
            #bidirectional=bidirectional
        ).to(device)
        self.sequential = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.Sigmoid()
        ).to(device)

         
        # END YOUR CODE
        
    def forward(self, text):

        
        '''
        In the forward() function, we decide which operations the input will undergo to generate the output.
        For example, in a sentiment classification model, the input usually goes through
        Embedding() -> RNN() -> Linear() in sequence to obtain the final output.
        '''
        # TO-DO 3-3: Determine how the model generates output based on input
        # BEGIN YOUR CODE
        embedded = self.embedding(text)  # Shape: (seq_length, batch_size, embedding_dim)
        rnn_output, _ = self.rnn(embedded)  # Shape: (seq_length, batch_size, hidden_size * num_directions)
        rnn_output = rnn_output[-1, :, :]  # Take the last hidden state
        output = self.sequential(rnn_output)  # Shape: (batch_size, 512)
        return output
        # END YOUR CODE
    
class RNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        
        '''
        Simply pass any parameters you want to initialize your model.
        Of course, if you want to hardcode all the parameters in the definition of the 
        YourModel class, then you don't need to pass any parameters here.
        '''
        # TO-DO 3-1: Pass parameters to initialize model
        self.config = config
        self.model = YourModel(
            # BEGIN YOUR CODE
            embedding_dim=500,
            num_embeddings=500, 
            num_layers=2,
            optimizer_type= "Adam",
            #loss_function="cross_entropy",
            learning_rate=0.01,
            #regularization=,
            hidden_size=500,
            dropout=0.18
            # END YOUR CODE
        ).to(config['device'])

    def forward(self, text):
        return self.model(text)
