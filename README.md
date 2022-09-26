# vice_headlines
Generating headlines for the Vice Youtube channel using a RNN implemented in Pytorch.

### Disclaimer
After experimenting for a while, I have come to the conclusion, that the available data may be insufficient to train a RNN from scratch.
I have decided to finetune BLOOM Model to learn more about the training process.  
You can find the BLOOM repo [here](https://github.com/marcderbauer/bloom).

<br/>
Update 23.09.2022 -- Restricting the output space could help. Meaning I would need to remove all the words from the embedding, which weren't in the input vocab. Not perfect, but would improve results. Also will need to change my generation strategy. More on this in issues
