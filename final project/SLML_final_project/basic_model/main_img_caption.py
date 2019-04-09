#############################################################
#                           Setup
# -----------------------------------------------------------        

import numpy as np
import os
import tensorflow as tf
import json

import load_data
import Function

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer

### set your paths ###
path_to_file = os.path.join(r"F:\Flickr30k")
path_to_feature = os.path.join(r"F:\Flickr30k","resnet101_fea")

# load annotation
data_cap, data_dic = load_data.load_caption(path_to_file)

############################################################
#                 Processing the data
# ----------------------------------------------------------

# ---------- split the train, valid, test dataset-----------
data_size = len(data_cap) # 31783

# check data_dic['iamges'], we would have that:
# - train = 29000
# - valid = 1014
# - test = 1000
# - rest = 769

# chech the number of each type of example
data_split_size = {'train':0, 'val':0, 'test':0, 'rest':0} # four type of data label

# count data_split_size
for i in range(data_size): 
    data_split_size[ data_dic['images'][i]['split'] ] += 1

print(data_split_size)

# split into train/val/test
def split_data(data_cap, data_dic, type = 'train'):
    split_cap = []  # extracted from data_cap
    split_id = []   # the order position 
    count = 0
    for i in range(data_size):
        if data_dic['images'][i]['split'] == type:
            split_cap.append(data_cap[i])
            split_id.append(i)
            count += 1
    return split_cap, split_id

train_cap, train_id = split_data(data_cap, data_dic, type = 'train')
valid_cap, valid_id = split_data(data_cap, data_dic, type = 'val')
test_cap, test_id = split_data(data_cap, data_dic, type = 'test')


# -------------------- Form the vocabulary ----------------------
word_freq = {} # calculate word frequency
for captionlist in train_cap: # in word_freq
    for capdict in captionlist:
        for word in (capdict['caption']):
            pass
            word_freq[word] = word_freq.get(word, 0) + 1

vocab = set(word_freq.keys()) # unique words
vocab_size = len(vocab) #8636
len(data_dic['wtol']) #8638
# word_freq.keys() is almost data_dic['wtol']. 
# two more words in data_dic['wtol'] are: marbles, 9-11, they are not in training set

# the word <-> ix method given by pf. fu's dataset (but we can use keras' api instead)
word2ix = data_dic['wtol']          # <dictionary> 'word': ix
ix2word = data_dic['ix_to_word']    # <dictionary> ix: 'word'

# *** there are still key 'idx' in data_cap and 'wtod' in data_dic, but I haven't find out what they mean

#--------------------- Prepare the feature-----------------------

# fetch the fc feature, form a (31783, 2048) size matrix 
# in the order of data_dic

fea_fc_size = 2048

path_of_stacked_fea = os.path.join(r'F:\Flickr30k','resnet101_fea','fea_fc.npy')

fea_fc = np.load(path_of_stacked_fea)

fea_fc_train = fea_fc[train_id]

transfer_values_size = fea_fc_size
# now fea_fc is a numpy array of shape (31783, 2048), stacking the features together

#################################################################
#                         Tokenizer
#----------------------------------------------------------------

# *** instead of wtol, here defines a specific tokenizer method

# now need to reform our captions for the Tokenizer from tensorflow
mark_start = 'ssss '
mark_end = ' eeee'

# eliminate the dictionary structure, only extract the 'caption' part
train_cap_join = [ [' '.join(cap['caption']) for cap in caplist] for caplist in train_cap] 

# mark the caption with start and end token
train_cap_marked = [ [mark_start + cap + mark_end for cap in caplist] for caplist in train_cap_join ] # mark <EOS> end token

# join all the word together in a single list, for the convinience of Tokenizer
train_cap_flat = [word for caplist in train_cap_marked for word in caplist]

num_words=8000 # *** how does the word count threshhold effect the performance???????????
# *** do we need to eliminate low-freq words?

# I don't understand the following now; what's tf tokenizer class?????????????????
class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""
    
    def __init__(self, texts, num_words=None):
        """
        :param texts: List of strings with the data-set.
        :param num_words: Max number of words to use.
        """

        Tokenizer.__init__(self, num_words=num_words)

        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""

        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        
        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text
    
    def captions_to_tokens(self, captions_listlist):
        """
        Convert a list-of-list with text-captions to
        a list-of-list of integer-tokens.
        """
        
        # Note that text_to_sequences() takes a list of texts.
        tokens = [self.texts_to_sequences(captions_list)
                  for captions_list in captions_listlist]
        
        return tokens

tokenizer = TokenizerWrap(texts=train_cap_flat, num_words=num_words)

# ok let's check the result of token
token_start = tokenizer.word_index[mark_start.strip()]
token_end = tokenizer.word_index[mark_end.strip()]

tokens_train = tokenizer.captions_to_tokens(train_cap_marked)


#################################################################
#                         Batching the data
#----------------------------------------------------------------

def get_random_caption_tokens(idx):
    """
    Given a list of indices for images in the training-set,
    select a token-sequence for a random caption,
    and return a list of all these token-sequences.
    """
    
    # Initialize an empty list for the results.
    result = []

    # For each of the indices.
    for i in idx:
        # The index i points to an image in the training-set.
        # Each image in the training-set has at least 5 captions
        # which have been converted to tokens in tokens_train.
        # We want to select one of these token-sequences at random.

        # Get a random index for a token-sequence.
        j = np.random.choice(len(tokens_train[i]))

        # Get the j'th token-sequence for image i.
        tokens = tokens_train[i][j]

        # Add this token-sequence to the list of results.
        result.append(tokens)

    return result

def batch_generator(batch_size):
    """
    Generator function for creating random batches of training-data.
    
    Note that it selects the data completely randomly for each
    batch, corresponding to sampling of the training-set with
    replacement. This means it is possible to sample the same
    data multiple times within a single epoch - and it is also
    possible that some data is not sampled at all within an epoch.
    However, all the data should be unique within a single batch.
    """

    # Infinite loop.
    while True:
        # Get a list of random indices for images in the training-set.
        idx = np.random.randint(data_split_size['train'],
                                size=batch_size)
        
        # Get the pre-computed transfer-values for those images.
        # These are the outputs of the pre-trained image-model.
        transfer_values = fea_fc_train[idx]

        # For each of the randomly chosen images there are
        # at least 5 captions describing the contents of the image.
        # Select one of those captions at random and get the
        # associated sequence of integer-tokens.
        tokens = get_random_caption_tokens(idx)

        # Count the number of tokens in all these token-sequences.
        num_tokens = [len(t) for t in tokens]
        
        # Max number of tokens.
        max_tokens = np.max(num_tokens)
        
        # Pad all the other token-sequences with zeros
        # so they all have the same length and can be
        # input to the neural network as a numpy array.
        tokens_padded = pad_sequences(tokens,
                                      maxlen=max_tokens,
                                      padding='post',
                                      truncating='post')
        
        # Further prepare the token-sequences.
        # The decoder-part of the neural network
        # will try to map the token-sequences to
        # themselves shifted one time-step.
        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]

        # Dict for the input-data. Because we have
        # several inputs, we use a named dict to
        # ensure that the data is assigned correctly.
        x_data = \
        {
            'decoder_input': decoder_input_data,
            'transfer_values_input': transfer_values
        }

        # Dict for the output-data.
        y_data = \
        {
            'decoder_output': decoder_output_data
        }
        
        yield (x_data, y_data)

batch_size = 64 # very RAM-comsuming for high batch-size

# an example of our batch
generator = batch_generator(batch_size=batch_size)

batch = next(generator)
batch_x = batch[0]
batch_y = batch[1]

# computing steps_per_epoch: assuming we want go over each captions once (but actually not, due to randomness)
num_captions_train = [len(captions) for captions in train_cap_join]
total_num_captions_train = np.sum(num_captions_train)

#steps_per_epoch = int(total_num_captions_train / batch_size)
steps_per_epoch = 5


#################################################################
#                        Setup the network
#----------------------------------------------------------------

# the hidden state-size
state_size = 512

# the word-embedding size
embedding_size = 128

#------------------------ Setup the model------------------------

# the inputing transfer value (image input feature)
transfer_values_input = Input(shape=(transfer_values_size,), 
                              name='transfer_values_input')

# image feature -> hidden state, note that we have an tanh activation (perhaps to be compatible with word embedding)
decoder_transfer_map = Dense(state_size,
                             activation='tanh',
                             name='decoder_transfer_map')

# the input of word token
decoder_input = Input(shape=(None, ), name='decoder_input')

# word token -> hidden state
decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')

# 3 GRU
# there are certainly many options for GRU units
# * recurrent_initializer='glorot_uniform': the initialization method
# * stateful: I haven't understood how to use it yet

'''
OPTIONAL:
recurrent_initializer='glorot_uniform',
recurrent_activation='sigmoid',
'''

decoder_gru1 = GRU(state_size, name='decoder_gru1',
                   return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',
                   return_sequences=True)
'''
The GRU layers output a tensor with shape [batch_size, sequence_length, state_size], where each "word" is 
encoded as a vector of length state_size. We need to convert this into sequences of integer-tokens that 
can be interpreted as words from our vocabulary.

One way of doing this is to convert the GRU output to a one-hot encoded array. It works but it is extremely 
wasteful, because for a vocabulary of e.g. 10000 words we need a vector with 10000 elements, so we can 
select the index of the highest element to be the integer-token.

Note that the activation-function is set to linear instead of softmax as we would normally use for one-hot 
encoded outputs, because there is apparently a bug in Keras so we need to make our own loss-function, as 
described in detail further below.
'''
# GRU units output -> logits, which is an 'ecode of word'
decoder_dense = Dense(num_words,
                      activation='linear',
                      name='decoder_output')


#################################################################
#                    Build and Train the model
#----------------------------------------------------------------
'''
The decoder is built using the functional API of Keras, which allows more flexibility 
in connecting the layers e.g. to have multiple inputs. This is useful e.g. if you want 
to connect the image-model directly with the decoder instead of using pre-calculated 
transfer-values.

This function connects all the layers of the decoder to some input of transfer-values.
'''

# *** how does the following work? ***
def connect_decoder(transfer_values):
    # Map the transfer-values so the dimensionality matches
    # the internal state of the GRU layers. This means
    # we can use the mapped transfer-values as the initial state
    # of the GRU layers.
    initial_state = decoder_transfer_map(transfer_values)

    # Start the decoder-network with its input-layer.
    net = decoder_input
    
    # Connect the embedding-layer.
    net = decoder_embedding(net)
    
    # Connect all the GRU layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    # Connect the final dense layer that converts to
    # one-hot encoded arrays.
    decoder_output = decoder_dense(net)
    
    return decoder_output

decoder_output = connect_decoder(transfer_values=transfer_values_input)

decoder_model = Model(inputs=[transfer_values_input, decoder_input],
                      outputs=[decoder_output])

#------------------------ Define loss function------------------------
'''refer to loss_function_explaination for further explaination'''

def sparse_cross_entropy(y_true, y_pred):
    """
    Calculate the cross-entropy loss between y_true and y_pred.
    
    y_true is a 2-rank tensor with the desired output.
    The shape is [batch_size, sequence_length] and it
    contains sequences of integer-tokens.

    y_pred is the decoder's output which is a 3-rank tensor
    with shape [batch_size, sequence_length, num_words]
    so that for each sequence in the batch there is a one-hot
    encoded array of length num_words.
    """

    # Calculate the loss. This outputs a
    # 2-rank tensor of shape [batch_size, sequence_length]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire 2-rank tensor, we reduce it
    # to a single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean

# ------------------ Compile the Training Model ------------------

optimizer = RMSprop(lr=1e-3) # what if changing the learning rate?????????????????????

decoder_target = tf.placeholder(dtype='int32', shape=(None, None))

decoder_model.compile(optimizer=optimizer,
                      loss=sparse_cross_entropy,
                      target_tensors=[decoder_target])

# ---------------------setup the check points --------------------
## *** how does this work??***
checkpoint_dir = os.path.join(r'F:\Flickr30k','model_hvass','checkpoints')
num_trials = '1'
path_checkpoint = os.path.join(checkpoint_dir, num_trials, 'ckpt.keras') 
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      verbose=1,
                                      save_weights_only=True)

callback_tensorboard = TensorBoard(log_dir=os.path.join(checkpoint_dir, 'log'),
                                   histogram_freq=0,
                                   write_graph=False)

callbacks = [callback_checkpoint, callback_tensorboard]

# --------------------- load the check points --------------------
'''try:
    decoder_model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)'''

# ------------------------ train the model -----------------------
decoder_model.fit_generator(generator=generator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=3,
                            callbacks=callbacks)

#################################################################
#                    Generate captions
#----------------------------------------------------------------

temperature = 0.5

def generate_caption(image_fea, max_tokens=30):
    """
    Generate a caption for the image feature.
    The caption is limited to the given number of tokens (words).
    """
    # -----------------------------------setup the generation process
    # Process the image with the pre-trained image-model
    # to get the transfer-values.
    transfer_values = np.expand_dims(image_fea, axis=0)
    
    # Pre-allocate the 2-dim array used as input to the decoder.
    # This holds just a single sequence of integer-tokens,
    # but the decoder-model expects a batch of sequences.
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    # The first input-token is the special start-token for 'ssss '.
    token_int = token_start

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0

    # hyperparameter for word sampling
    temperature = 1

    # While we haven't sampled the special end-token for ' eeee'
    # and we haven't processed the max number of tokens.
    while token_int != token_end and count_tokens < max_tokens:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = \
        {
            'transfer_values_input': transfer_values,
            'decoder_input': decoder_input_data
        }

        # Note that we input the entire sequence of tokens
        # to the decoder. This wastes a lot of computation
        # because we are only interested in the last input
        # and output. We could modify the code to return
        # the GRU-states when calling predict() and then
        # feeding these GRU-states as well the next time
        # we call predict(), but it would make the code
        # much more complicated.
        
        # Input this data to the decoder and get the predicted output.
        decoder_output = decoder_model.predict(x_data)

        # Get the last predicted token as a one-hot encoded array.
        # Note that this is not limited by softmax, but we just
        # need the index of the largest element so it doesn't matter.
        token_onehot = decoder_output[0, count_tokens, :] # token_onehot: shape:(num_words,), type: nparray

        #--------------------------------------sample the next word
        # Convert to an integer-token.
        # ========= argmax ===========
        token_int = np.argmax(token_onehot)
        '''There are basically 2 ways to sample predicted word:
        1. sampling 2. beam search'''

        # ========= samplling ===========
        # token_onehot = token_onehot / temperature
        '''token_prob = Function.softmax(token_onehot, temperature)
        token_int = np.random.choice(range(num_words), p=token_prob)'''

        #------------------------------------------------------
        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer.token_to_word(token_int)

        # Append the word to the output-text.
        output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    # This is the sequence of tokens output by the decoder.
    output_tokens = decoder_input_data[0]

    # Plot the image.
    # plt.imshow(image)
    # plt.show()
    
    # Print the predicted caption.
    print("Predicted caption:")
    print(output_text)
    print()


# check the generation result
fea_val1 = fea_fc[valid_id[50]]

generate_caption(fea_val1, max_tokens=30)
