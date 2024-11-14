# Grapheme-to-Phoneme  

This is a model I created for my deep learning capable of taking phoneme inputs(how the word is spelled) and predicting the phonemes(how its pronounced).  

In the endevour to create the best model I implemented several different approaches such as:  
  * RNN (this was my baseline)
  * GRU
  * LSTM
  * LSTM variations with/withou Bi-Directional, Stacked, Self-Attention
  * Transformer

Ultimately I found the Bi-Directional Stacked LSTM to produce the best results.  
  

<div align="center">
  <img align="center" src="https://lukedojan.com/images/G2P_updated.gif" alt="Grapheme to Phoneme Conversion">
  <p style="text-align: center;"> 
    Figure 1: Sample output of model
  </p>
</div>
