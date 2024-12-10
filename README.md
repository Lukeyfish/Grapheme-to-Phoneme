# Grapheme-to-Phoneme Deep Learning Model

## Purpose and Scope
For my final project in my deep learning class at WWU, I was tasked with implementing a sequence-to-sequence model capable of doing grapheme-to-phoneme conversion (G2P). This project sits at the intersection of natural language processing, speech technology, and machine learning, capable of predicting phonemes(how the word is pronounced) based on the graphemes(how the word is spelled).  

## Approach and Implementation
### Model Architecture
In the pursuit to create the best model, I implemented several different architectures, including:  
1. Recurrent Neural Networks (RNN)
   * Served as my baseline
   * Provided a benchmark to compare against
2. Gated Recurrent Units (GRU)
   * Improved model performance
   * Better at handling long-range memory
3. Long Short-Term Memory Networks (LSTM)
   * Advanced sequential model with enhanced memory retention
   * Gradients vanish less easily
4. LSTM Variations
   * Bi-Directional: Processes sequences in both forward and backward directions
   * Stacked LSTM: Multiple LSTM layers for increased model complexity
   * Self-Attention Mechanisms: Improved context understanding and feature extraction
5. Transformer
   * SOTA model leveraging self-attention mechanism
   * Performs very well with seq-to-seq problems
     
## Performance Metric

Models were evaluated using the performance metric of Word Error Rate(WER), evaluated from:

$$
\text{WER} = \frac{S + D + I}{S + D + C}
$$

where:  

- **S** is the number of substitutions,  
- **D** is the number of deletions,  
- **I** is the number of insertions,  
- **C** is the number of correct words

## Results
In the end, my best model ended up being the Bi-Directional Stacked LSTM. This model achieved the lowest WER and produced the most coherent phonetic results. Figure 1 shows an example of what my model could produce.


<div align="center">
  <img align="center" src="https://lukedojan.com/images/G2P_updated.gif" alt="Grapheme to Phoneme Conversion">
  <p style="text-align: center;"> 
    Figure 1: Sample model output
  </p>
</div>
