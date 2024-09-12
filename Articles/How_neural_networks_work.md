# How AI Really Works: The Basics in Details

Everyone talks about AI today but [almost no one seems to understand how it works](https://www.reddit.com/r/MachineLearning/comments/r76igz/discussion_rant_most_of_us_just_pretend_to/), this is because there are very few sources that really go into detail and in any case, almost all are incomplete, after a lot of research and many calculations (too many) I finally managed to write complete notes on the operating principles that underlie modern artificial intelligence.

The field of AI is very broad, technically any program that tries to emulate human intelligence (or a subset of it) is considered artificial intelligence, but what most people mean by AI today are neural networks (more precisely transformers for modern LLMs).

In this article, I will explain what neural networks are and how their training makes them work, in particular, we will see the simplest neural network, the feed-forward network, which is the basis of how transformers work (maybe an article on how transformers themselves work will come in the future).

This is not a superficial article, as mentioned, I will not spare the details, so, to better understand what I will say, you will need to have a good knowledge of [multivariable calculus](https://en.wikipedia.org/wiki/Multivariable_calculus) and [matrices](https://en.wikipedia.org/wiki/Matrix_(mathematics)).

We are ready to start:

A **Feed Forward Network** (or **multilayer perceptron**) is the simplest type of neural network, that is, a network of artificial neurons, which is inspired by the structure of the neurons in the brain to create, through training, any function that, given certain inputs, produces certain outputs based on the data on which the training was done. 
Although modern neural networks use many optimization techniques for training and more complex neural network structures, the basic principles of their operation are based on (and often used) the concepts that I will explain in this article.
Before explaining the structure of a Feed Forward Network, let's see exactly what an artificial neuron is:

## Artificial Neuron

Given a real function of real var **σ**, called the **activation function**, n real values **​​W<sub>1</sub>**, **​​W<sub>2</sub>**, ..., **​​W<sub>n</sub>**, called **weights**, a real value **b** called **bias**:</br>
An **artificial neuron** (or **neuron**) is a scalar function of n real variables that receives as input n real values **​​x<sub>1</sub>**, **​​x<sub>2</sub>**, ..., **​​x<sub>n</sub>**, and produces a real value equal to the image, in σ, of the sum of ​​x<sub>1</sub>, ​​x<sub>2</sub>, ..., ​​x<sub>n</sub>, with each x<sub>i</sub> multiplied by the weight W<sub>i</sub>, and with b also added.</br>
Formally, given the premises above, an artificial neuron is the function specified as follows:

**neuron: D<sub>1</sub> x D<sub>2</sub> x ... x D<sub>n</sub> --> R, &nbsp;&nbsp;&nbsp;&nbsp; neuron(x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>)  =  σ(b + W<sub>1</sub> * x<sub>1</sub> + W<sub>2</sub> * x<sub>2</sub> +...+ W<sub>n</sub> * x<sub>n</sub>)**
