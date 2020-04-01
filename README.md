# Machine Learning for Health Informatics 2020 Assignment

## Introduction

This assignment requires you to build a simple text augmentation pipeline to create data for the training of a sentiment analysis model.

**This file will be updated as the course progresses! Check back here to see updates during the semester.**

## Augmentation

What is data augmentation?

Data augmentation is the creation of new data through the manipulation of an existing data set. Data augmentation is used extensively in image analysis and deep learning, as new image data can quite readily be created using basic techniques. For information regarding image augmentation, see <https://github.com/mdbloice/Augmentor>

Examples of image augmentation might include flipping images along their horizontal axes. Below is a simple demonstration of this: 

![](Images/augmentation-example1.jpg)

Imagine you had a data set of 100s of houses: by flipping them all on the horizontal axis, you have doubled your data set size.

The key to augmentation is that it must be **label preserving**. That is, after you have augmented your house images, they must still be recognisable and must still be representative of houses in the real world. For example, you would not flip images of houses on their vertical axes. Houses will never appear upside down in real life, and therefore this is not a label preserving transformation. Domain knowledge is therefore required for data augmentation.

Why do we do augmentation? When we train machine learning models, augmentation makes the models more robust and generalises better to new, unseen data.

## Text Augmentation

For this assignment, you will, however, be working on **text data** and **text augmentation**.

Just as in image augmentation, text augmentation creates new data from an existing data set. Some examples might be:

- Swapping word order
  - "the small brown fox" → "the brown small fox"
- Replacing digits with text:
  - "2 small brown foxes" → "two small brown foxes"
- Dictionary/thesaurus-based word replacement:
  - "the small brown fox" → "the slight brown fox"
- Purposely misspelling words:
   - "the small brown fox" → "the smll brown fox"
- Etc.

Text data is much different than image data, however. It is more challenging to create augmentation techniques that are label preserving, due to the contextual nature of written language.

However, there are a number of techniques which may be universally applicable, and that is the focus of the work for this assignment.

Also, in text data there is the concept of a **corpus**, which is made up of **documents**, which consist of **paragraphs**, that consist of **sentences**, then **words**, then individual **characters**. 

However, for this assignment, we will use a simple text data set, where we will not need to worry much about paragraphs, documents, etc. We discuss the data set in more detail below.

## Task

The task for this assignment therefore, is to augment a text data set and train a model using your augmented data set, comparing it to the baseline of no augmentation. 

### Data Set
The text data that we will use for this assignment is the Sentiment140 data set:

<http://help.sentiment140.com/for-students/>

The ZIP file is 78MB in size, but its uncompressed size is 234MB. The ZIP contains two files: a training set, with 1.6 million tweets, and a test set. For this assignment, please ignore the test set in this ZIP file. You will create your own test set for this assignment.

This is a binary classification problem. Each tweet in the training set is classified as either being negative (0) or positive (4) in sentiment: 

| Class | ID | Timestamp | Query | Handle | Tweet |
|---|---|---|---|---|---|
| 0 | 1467814438 | Mon Apr 06 22:20:44 PDT 2009 | NO_QUERY | ChicagoCubbie | I hate when I have to call and wake people up |
| 0 | 1467814783 | Mon Apr 06 22:20:50 PDT 2009 | NO_QUERY | KatieAngell | Just going to cry myself to sleep after watching Marley and Me. |
| 0 | 1467814180 | Mon Apr 06 22:20:40 PDT 2009 | NO_QUERY | viJILLante | this week is not going as i had hoped |
| 0 | 1467814192 | Mon Apr 06 22:20:41 PDT 2009 | NO_QUERY | Ljelli3166 | blagh class at 8 tomorrow |
| 4 | 2174491864 | Sun Jun 14 22:39:59 PDT 2009 | NO_QUERY | roachls | just the lil things.. can make my day |
| 4 | 2174490774 | Sun Jun 14 22:39:50 PDT 2009 | NO_QUERY | HelloLizzi | I'm a lonerrrr  this is the life. |
| 4 | 2174489472 | Sun Jun 14 22:39:40 PDT 2009 | NO_QUERY | RachelBrammer | Another scary movie with the girls  CWH(: |
| 4 | 2174504162 | Sun Jun 14 22:41:33 PDT 2009 | NO_QUERY | kenbank | Thougjht I'd shove a tweet out to all my followers! You Rock! |

*Note: the test set contains a third, neutral class (2). We will ignore this class for this assignment.*

It is your task to create a model to predict either a negative (0) or positve (4) sentiment. You can test your model's accuracy using a held back portion of the data - the test set.

### Creating the Test Set

Normally, I would suggest you split the data set in to three: a training set, a validation set, and a test set. You would use the validation set to tune your hyper-parameters and model parameters. When you have found a model that performs well, test your final model on the test set. It is normally good advice to not use your test set to tune your model, or you will over fit your parameters!

For this assignment however, we will use only a training set and a test set.

Choose a random 10% of the tweets as your test set. That is 160,000 tweets. Use the remaining 1.44 million tweets to train your model. 

You may want to ensure that you have approximately the same number of positive and negative classes in the test set that you choose. This makes it easier to interpret common accuracy and performance metrics.

If you find this amount of data too large to handle for the resources you have, then reduce the size of the training set and test set proportionally. You could use 160,000 tweets in total, with 144,000 used for training and 16,000 used for testing, for example.

You will not be able to open the training set using Excel or LibreOffice. Both have a limit of approximately 1,000,000 rows. I would recommend using Visual Studio Code, as it can handle very large files quite easily and you can also search large files without difficulty: 

<https://code.visualstudio.com/>

So, to recap: ignore the test set provided in the ZIP file on the Sentiment140 website. Create your own test set by randomly selecting 10% of the training set. Try to ensure this is balanced (equal number of negative and positive tweets).

## Tools and Libraries

For this assignment we will use Python. I would suggest using a tool such as NLTK (The Python Natural Language Toolkit) for reading the data and performing some standard operations on the data, such as stop word removal:

<https://www.nltk.org/>

Install via pip:

```
pip install nltk 
```

For training the model, I would suggest SciKit Learn. Most types of machine learning algorithms are supported by SciKit Learn:

<https://scikit-learn.org/>

Installation is also via pip:

```
pip install scikit-learn
```

Random Forests may be a good algorithm to try first. 

I would also suggest you install Pandas and NumPy.

Other NLP (Natural Language Processing) packages exist which you may prefer to use instead of NLTK, such as TextBlob:

<https://textblob.readthedocs.io>

As inspiration, you can follow this example, where NLTK was used with SciKit Learn to build a simple classifier:

<http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/>

You could use a similar procedure, but you must also integrate augmentation in to your approach.

## How to Implement Text Augmentation

How you implement text augmentation is up to you. You may want to utilise NLTK or some other NLP (Natural Language Processing) library, or you may want to augment the data set “offline” before you read it in using NLTK and pass it to the algorithm for training.

However, for this assignment, you will be required to use **at least four different text augmentation techniques**. Which four you choose is up to you. As mentioned previously, not all techniques make sense for all data sets or classification problems. A thesaurus based approach may well make sense for this data set, for example. NLTK allows you to use WordNet to find similar words for a given word. WordNet is not really a thesaurus, however it may be useful to begin your assignment (if you decide that want to use word replacement as an augmentation technique).

## How to Proceed

### Stage 1

**The deadline for Stage 1 of this assignment is the 30th of April, 2020.**

The following are the requirements for Stage 1 of this assignment:

1. Download the data set from the Semantic140 website
2. Examine the data, learn what each column in the training set CSV file mean, etc.
3. Research and install the various tools you might use to perform the assignment, such as NLTK, TextBlob, and SciKit Learn. 
4. Create your training set / test set split (10% of the data should be used for testing)
5. Decide on four text augmentation techniques you think make sense for this data set

### Stage 2

Stage 2 of the assignment will be updated in due course. However, in this stage of the experiment you will train a model to perform the sentiment classification of your test set. You will compare a baseline (without augmentation) to a model trained with text augmentation. In this stage you will need to pre-process the data set, and decide how you will implement text augmentation on the pre-process data set. What does pre-processing mean? For example you may want to replace all timestamps with either 0 for daytime or 1 for night time.

*This section will be updated in due course.*

## Contact 

For questions, please simply create issues in this repository:

<https://github.com/human-centered-ai-lab/MLHI-2020/issues> 

*Remember: the goal of this experiment/assignment is not to create a world-beating sentiment analysis model. The goal is to use text augmentation and gauge its effectiveness at improving a baseline result without augmentation. Also, it is an opportunity to be creative in choosing text augmentation technique or inventing a novel text augmentation technique.*
