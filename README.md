# Feature Selection in NLP using Evolutionary Algorithms


## Overview
Selecting good features is one of the most important challenge while designing a machine learning model. The task of feature selection especially becomes more trick while selecting features that are extracted from text data. In this project we wanted develop an algorithm which good automate the process of feature selection and output the best set of features on which machine learning models can run. The best way to select a subset of features from the complete set is buy first analysing every unique subsets performance, However this very time consuming and with such 32 features 10^9 combinations can be formed. We thus moved in the direction of evolutionary algorithms, which are known to find the approximate global optimum while take very less time in comparision to aformentioned method.

## Evolutinary algorithm based feature selection
#### Prerequistes
In order to fully understand the working of Evolutinary algorithm based feature selection. It is important to know what are evolutionary algorithms.Evolutionary algorithms are based on concepts of biological evolution. A 'population' of possible solutions to the problem is first created with each solution being scored using a 'fitness function' that indicates how good they are. The population evolves over time and (hopefully) identifies better solutions.<br>
In our project we mainly use the following two evolutionary Algorithms- <br>
* Particle Swarm optimisation algorithm - In computational science, particle swarm optimization (PSO)[1] is a computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. It solves a problem by having a population of candidate solutions, here dubbed particles, and moving these particles around in the search-space according to simple mathematical formula over the particle's position and velocity. Each particle's movement is influenced by its local best known position, but is also guided toward the best known positions in the search-space, which are updated as better positions are found by other particles. This is expected to move the swarm toward the best solutions.
* Genetic Algorthm - Genetic algorithms are randomized search algorithms that have been developed in an effort to imitate the mechanics of natural selection and natural genetics. Genetic algorithms operate on string structures, like biological structures, which are evolving in time according to the rule of survival of the fittest by using a randomized yet structured information exchange. Thus, in every generation, a new set of strings is created, using parts of the fittest members of the old set<br>
#### Our Technique
We experimented with combining both pso and G.A. Since each offered its own set of advantages. A major problem we observed while experimenting with genetic algorithm is that it works very well and reaches global optimum when initialised with a good population. We thus run PSO first to get a good population, Then genetic algorithm is intialised with this population. The algorithm followed is as follows- <br>

![ALgorithm](https://github.com/AnunayGupta/Feature-Selection-in-NLP-through-Evolutionary-Optimisation-Algorithms/blob/34b101dbf151fcec1e7bf70cd594a9e8cafb6b29/Static/Screenshot%202021-07-18%20at%201.27.37%20AM.png)
![ALgorithm](https://github.com/AnunayGupta/Feature-Selection-in-NLP-through-Evolutionary-Optimisation-Algorithms/blob/34b101dbf151fcec1e7bf70cd594a9e8cafb6b29/Static/Screenshot%202021-07-18%20at%201.27.48%20AM.png)
![ALgorithm](https://github.com/AnunayGupta/Feature-Selection-in-NLP-through-Evolutionary-Optimisation-Algorithms/blob/34b101dbf151fcec1e7bf70cd594a9e8cafb6b29/Static/Screenshot%202021-07-18%20at%201.27.56%20AM.png)


#### Datasets
* [Jruvika](https://www.kaggle.com/jruvika/datasets)
* [Liar](https://github.com/thiagorainmaker77/liar_dataset)
* [Buzzfeed](https://www.kaggle.com/sohamohajeri/buzzfeed-news-analysis-and-classification)
* [Covid](https://competitions.codalab.org/competitions/26655)

![Dataset balanced visualisation](https://github.com/AnunayGupta/Feature-Selection-in-NLP-through-Evolutionary-Optimisation-Algorithms/blob/e5092b845fe66e5eda27c6a847dcaa3e8208928b/Static/Screenshot%202021-07-18%20at%201.53.37%20AM.png "This is a sample image.")


#### Types of features experimented with
* Word2Vec - is a two-layer neural net that processes text by “vectorizing” words. Its input is a text corpus and its output is a set of vectors: feature vectors that represent words in that corpus. While Word2vec is not a deep neural network, it turns text into a numerical form that deep neural networks can understand.
* Doc2Vec -  is an NLP tool for representing documents as a vector and is a generalizing of the word2vec method.
* TF-IDF - is a statistical measure that evaluates how relevant a word is to a document in a collection of documents. This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents

#### Resuts
