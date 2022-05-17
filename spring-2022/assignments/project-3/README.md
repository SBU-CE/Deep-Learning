# Assignment #3 - Transformers and BERT/ParsBERT


Deep Learning / Spring 1401, Shahid Beheshti University


<br><br>

**Please pay attention to these notes:**
<br>


- If you need any additional information, please review the assignment page on the course github.
- The items you need to answer are highlighted in red and the coding parts you need to implement are denoted by:
```
########################################
#              Your Code               #
########################################
```

- Finding any sort of copying will zero down that assignment grade.
- If you have any questions about this assignment, feel free to ask us.
- You must run this notebook on Google Colab platform, it depends on Google Colab VM for some of the depencecies.

<br><br>
## BERT
BERT stands for Bi-directional Encoder Representation from Transformers is designed to pre-train deep bidirectional representations from unlabeled texts by jointly conditioning on both left and right context in all layers. The pretrained BERT model can be fine-tuned with just one additional output layer (in many cases) to create state-of-the-art models. This model can use for a wide range of NLP tasks, such as question answering and language inference, and so on without substantial task-specific architecture modification.

![BERT INPUTS](https://res.cloudinary.com/m3hrdadfi/image/upload/v1595158991/kaggle/bert_inputs_w8rith.png)

As you may know, the BERT model input is a combination of 3 embeddings.
- Token embeddings: WordPiece token vocabulary (WordPiece is another word segmentation algorithm, similar to BPE)
- Segment embeddings: for pair sentences [A-B] marked as $E_A$ or $E_B$ mean that it belongs to the first sentence or the second one.
- Position embeddings: specify the position of words in a sentence

<br><br>
Before going more further into code, let us introduce ParsBERT.
<br>

## ParsBERT
ParsBERT is a monolingual language model based on Google's BERT architecture. This model is pre-trained on large Persian corpora with various writing styles from numerous subjects (e.g., scientific, novels, news, ...) with more than 3.9M documents, 73M sentences, and 1.3B words. For more information about ParsBERT, please check out the article: [DOI: 10.1007/s11063-021-10528-4](https://link.springer.com/article/10.1007/s11063-021-10528-4)

<br>
So, now you have a little understanding of BERT in total, we need to know how to use ParsBERT in our project. In this assignment, you will implement a fine-tuned model on the Sentiment Analysis task for PyTorch. Good Luck!


<br><br>

**Setup**
- Download assignment3.ipynb to obtain the assignment jupyter notebook.
- Go to https://colab.research.google.com/.
- Switch to Upload tab, choose assignment3.ipynb and click upload.
- Now You’re ready to go.
