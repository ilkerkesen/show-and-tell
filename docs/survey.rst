Literature Survey
===================

Show and Tell: A Neural Image Caption Generator 
--------------------------------------------------
It is a simple end-to-end model :cite:`vinyals2015show` which is built on top of a Convolutional Neural Network (CNN) and a special type of Recurrent Neural Network called as Long Short Term Memory Network (LSTM) . At initial time step visual are used in LSTMs as input. After first time step, vector representations of sentence words becomes input parameters of recurrent network. Word embeddings are also learned during training. Sampling and BeamSearch methods are used for inference. The proposed framework is tested on five different public datasets and it was the state-of-art by then. Generation and ranking results are reported using several evaluation metrics.

.. figure:: static/show-and-tell.png
   :align: center
   :scale: 100%
   :alt: Model Overview

   Model Overview

Long-term Recurrent Convolutional Networks for Visual Recognition and Description
-------------------------------------------------------------------------------------
This paper :cite:`donahue2015long` presents solutions for image captioning (as well as video captioning and activity recognition). Its framework is also a combination of a CNN and LSTMs. Unlike :cite:`vinyals2015show`, visual features are used in each time step. Also, it proposes "factorization" property for LSTM networks and makes experiments with three different models. It uses a different inference strategy (I didn't figured out it clearly yet) together with BeamSearch. The results are competitive. What's more, it has `Caffe implementation <https://github.com/BVLC/caffe/pull/2033/commits/668b17ede1e31a1d4a2663bd81357ab92065f812>`_.

.. figure:: static/lrcn.png
   :align: center
   :scale: 100%
   :alt: Model Overview

   Model Overview

.. bibliography:: survey.bib
