from __future__ import absolute_import, division, print_function

import os
from six import moves
import ssl
import tflearn
from tflearn.data_utils import *

path = "IndianNames.txt"

maxlen = 20

X, Y, char_idx = \
    textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3)

g = tflearn.input_data(shape=[None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 512, return_seq=True)
#converging on identitcal positions
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.01)

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='model_Indian_Names')

#$training
for i in range(40):
    seed = random_sequence_from_textfile(path, maxlen)
    m.fit(X, Y, validation_set=0.1, batch_size=128,
          n_epoch=1, run_id='Indian_Names')
    print("-- Testing--")
    print(m.generate(30, temperature=1.0, seq_seed=seed))

    #model = DNN(network, tensorboard_verbose=3)

  #To visualize data, use this command on terminal
  #tensorboard --logdir='/tmp/tflearn_logs'
