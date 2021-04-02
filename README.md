
# GoEmotions : text-based emotion detection

## Introduction
Nowadays, virtual interactions occupy an important place in our lives, especially in the COVID-19 context.

We wanted to address the problem of text-based emotion detection as it is really difficult to interpret emotions in a text message or any social media comment. There is always a lot of subjectivity in the way sentences are turned (sarcasm, irony, etc), in addition to the explosion of emojis use.

Some of the applications are: social media analysis, detection of emotional distress, etc.

## Objective
Although this topic has been tackled many times, we noticed that the spectrum of emotions is generally limited to a small number (6 to 12).

The objectives of this study were:

 - Find a large dataset of text samples labeled using a large spectrum of emotions
 - Train text classification models using Machine Learning and NLP techniques

## Dataset
For this study, we used the GoEmotions dataset [1].

This dataset was built by a Google Research team and gathers **more than 58K Reddit english comments**. In fact, it is **the largest manually labeled dataset** in this topic.

However, this dataset presents several challenges:

 - **Very high number of emotions** to detect: 27 emotions + "neutral"
 - **Class imbalance**: ~30% of "neutral" samples
 - **Multi-label**: each sample can be labeled with up to 5 different emotions

## Prerequisites
### Environment 
The notebooks were mainly developed using Google Colaboratory because of the availability of GPU computing resources. However, we highly recommend to use GPU with parsimony as you can quickly reach free usage limitations. Please refer to this [documentation](https://colab.research.google.com/notebooks/gpu.ipynb) to see how to enable/disable GPU for a Google Colaboratory notebook.

### Dependencies

 - Code was written in Python 3.
 - The python packages and the versions can be found in the file "requirements.txt" and can be installed using `pip`.
 
## Usage
For the sake of clarity and efficiency, we decided to split the work in multiple notebooks. 

We kindly encourage you to read them in the following order to get a better grasp of what has been done.

Also, kindly modify the different paths in the notebooks before executing.

### 1_EDA_Preprocessing.ipynb
 **Exploration and cleaning of the data in preparation for multi-label text classification tasks**
 
 - Input files
	 - train.tsv: train dataset
	 - dev.tsv: validation dataset
	 - test.tsv: test dataset
	 - emotions.txt: list of emotions in the GoEmotions taxonomy
 - Output files
	 - train_clean.csv: clean train dataset
	 - val_clean.csv: clean validation dataset
	 - test_clean.csv: clean test dataset

### 2_Baseline_Modeling.ipynb
 **Creating baseline models for emotion detection**
 
 - Input files
	 - train_clean.csv: clean train dataset
	 - val_clean.csv: clean validation dataset
	 - test_clean.csv: clean test dataset
	 - emotions.txt: list of emotions in the GoEmotions taxonomy
 
### 3_BERT_Model_1.ipynb
 **Fine-tuning a BERT model for emotion detection**
 
 - Input files
	 - train_clean.csv: clean train dataset
	 - val_clean.csv: clean validation dataset
	 - test_clean.csv: clean test dataset
	 - emotions.txt: list of emotions in the GoEmotions taxonomy
	 - ekman_labels.txt: list of emotions in the Ekman taxonomy

###  4_BERT_Model_2.ipynb
 **Fine-tuning a BERT model for emotion detection (without "neutral" samples) **
 
 - Input files
	 - train_clean.csv: clean train dataset
	 - val_clean.csv: clean validation dataset
	 - test_clean.csv: clean test dataset
	 - emotions.txt: list of emotions in the GoEmotions taxonomy
	 - ekman_labels.txt: list of emotions in the Ekman taxonomy
 - Output files
	 - bert-weights.hdf5: model weights

### my-annoying-shrink-app :) 
This is a small and funny web application that showcases our work.

It mimics a therapist that answers his patients with irritating replies, only according to the emotions he detected after he asked the question: *"how are you feeling today ?"* 

Each time the patient (you) enters a text, the therapist (our model) analyzes the emotions with their probabilities, and and the app returns these in a web interface together with a predefined "answer" for each emotion.

 1. Make sure to install all the packages in "requirements.txt"
 2. Open your terminal and navigate to the my-annoying-shrink-app directory
 3. run the following command: `streamlit run app.py`

## Team contributors 
This work was the Capstone Project as part of a 12-weeks intensive data science program (Jedha Bootcamp) my teammates and followed:

Perrine Panisset, Florian Akretche, Ibrahim Benjelloun

## References
[1] GoEmotions: A Dataset of Fine-Grained Emotions. Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen], Gaurav Nemade, Sujith Ravi. [arXiv:2005.00547v2](https://arxiv.org/abs/2005.00547v2)
