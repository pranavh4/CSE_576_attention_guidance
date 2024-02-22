# Attention Guidance: Transformer Augmentations for the Inverse Scaling Problem

Large Language Models exhibit worsening performance with an increase in model
size for certain task types, which is known as the [Inverse Scaling Problem](https://github.com/inverse-scaling/prize). 
We implement an augmentation to the transformer architecture to guide it to
pay attention to certain crucial linguistic features. We find that the augmented
transformer model does perform better on certain inverse scaling tasks whose data
includes these linguistic features. However, the gains are quite minimal and we
find that the inverse scaling problem is not entirely solved, even though the performance of the models improved.

The full report with our findings can be found [here](/Attention_Guidance_Inverse_Scaling.pdf) 
