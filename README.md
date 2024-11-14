# Grapheme-to-Phoneme  

This is a model I created for my deep learning capable of taking phoneme inputs(how the word is spelled) and predicting the phonemes(how its pronounced).  

In the endevour to create the best model I implemented several different approaches such as:  
  * RNN (this was my baseline)
  * GRU
  * LSTM
  * LSTM variations with/withou Bi-Directional, Stacked, Self-Attention
  * Transformer

Ultimately I found the Bi-Directional Stacked LSTM to produce the best results.  
  
Heres a little sample of what the model could output: 
### Grapheme ->      Phoneme  
* EXAM       ->  EH1 K S AH0 M  
* MUSSELS    ->  M AH1 S AH0 L Z    
* PANASONIC  ->  P AE2 N AH0 S OW1 N IH0 K  
* ACCREDITED ->  AH0 K R EH1 D AH0 T IH0 D  
