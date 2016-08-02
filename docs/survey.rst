Literature Survey
===================

Show and Tell: A Neural Image Caption Generator 
--------------------------------------------------
It is a simple end-to-end model :cite:`vinyals2015show` which is built on top of a Convolutional Neural Network (CNN) and a special type of Recurrent Neural Network called as Long Short Term Memory Network (LSTM) . While CNN's hidden layer is connected to LSTM as hidden state, vector representations of sentence words are connected to LSTM network through input gates. Word embeddings are also learned during training. Sampling and BeamSearch methods are used for inference. The proposed framework is tested on five different public datasets and it was the state-of-art by then. Generation and ranking results are reported using several evaluation metrics.

.. figure:: static/show-and-tell.png
   :align: center
   :scale: 100%
   :alt: Model Overview

   Framework Overview

.. bibliography:: survey.bib
