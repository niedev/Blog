# How AI Really Works: The Basics in Details

Everyone talks about AI today but [almost no one seems to understand how it works](https://www.reddit.com/r/MachineLearning/comments/r76igz/discussion_rant_most_of_us_just_pretend_to/), this is because there are very few sources that really go into detail and in any case, almost all are incomplete, after a lot of research and many calculations (too many) I finally managed to write complete notes on the operating principles that underlie modern artificial intelligence.

The field of AI is very broad, technically any program that tries to emulate human intelligence (or a subset of it) is considered artificial intelligence, but what most people mean by AI today are neural networks (more precisely transformers for modern LLMs).

In this article, I will explain what neural networks are and how their training makes them work, in particular, we will see the simplest neural network, the feed-forward network, which is the basis of how transformers work (maybe an article on how transformers themselves work will come in the future).

This is not a superficial article, as mentioned, I will not spare the details, so, to better understand what I will say, you will need to have a good knowledge of [multivariable calculus](https://en.wikipedia.org/wiki/Multivariable_calculus) and [matrices](https://en.wikipedia.org/wiki/Matrix_(mathematics)).

We are ready to start:

A **Feed Forward Network** (or **multilayer perceptron**) is the simplest type of neural network, that is, a network of artificial neurons, which is inspired by the structure of the neurons in the brain to create, through training, any function that, given certain inputs, produces certain outputs based on the data on which the training was done. 
Although modern neural networks use many optimization techniques for training and more complex neural network structures, the basic principles of their operation are based on (and often used) the concepts that I will explain in this article.
Before explaining the structure of a Feed Forward Network, let's see exactly what an artificial neuron is:
</br></br>

## Artificial Neuron

Given a real function of real var **σ**, called the **activation function**, n real values **​​W<sub>1</sub>**, **​​W<sub>2</sub>**, ..., **​​W<sub>n</sub>**, called **weights**, a real value **b** called **bias**:</br>
An **artificial neuron** (or simply **neuron**) is a [scalar function of n real variables](https://en.wikipedia.org/wiki/Function_of_several_real_variables) that receives as input n real values **​​x<sub>1</sub>**, **​​x<sub>2</sub>**, ..., **​​x<sub>n</sub>**, and produces a real value equal to the image, in σ, of the sum of ​​x<sub>1</sub>, ​​x<sub>2</sub>, ..., ​​x<sub>n</sub>, with each x<sub>i</sub> multiplied by the weight W<sub>i</sub>, and with b also added.</br>
Formally, given the premises above, an artificial neuron is the function specified as follows:

**neuron: D<sub>1</sub> x D<sub>2</sub> x ... x D<sub>n</sub> --> R, &nbsp;&nbsp;&nbsp;&nbsp; neuron(x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>)  =  σ(b + W<sub>1</sub> * x<sub>1</sub> + W<sub>2</sub> * x<sub>2</sub> +...+ W<sub>n</sub> * x<sub>n</sub>)**

#### Artificial vs biological neurons:

I'm not a biologist, so take this part with a grain of salt:

Artificial neurons are inspired by biological ones, they are a simpler representation of them, but they should work more or less on the same principles. Although it is still not known exactly how a single biological neuron functions, by combining several artificial neurons it is possible to emulate its behavior (see [this paper](https://www.quantamagazine.org/how-computationally-complex-is-a-single-neuron-20210902/)).
</br></br>

## Definition of Feed Forward Network

**A <code>feed forward network</code> is a function that receives as input n<sub>0</sub> real values ​​x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n0</sub>, an activation function σ, and a set θ of matrices and tuples** (containing the weights and biases of each neuron in the network) **and produces as output n<sub>L</sub> real values ​​y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>nL</sub> depending on all inputs, more precisely the law f is a law defined through the combination of [artificial neurons](https://github.com/niedev/Blog/edit/main/Articles/How_neural_networks_work.md#artificial-neuron) organized in a network-like way, structured in this way:**

![image](https://github.com/user-attachments/assets/a44445fe-f75a-4e29-90ae-696af455ddc7)

<img align="right" width="500" src="https://github.com/user-attachments/assets/672c2d81-e487-4344-85ef-c232b4f19b87">

</br></br>
The network is divided into levels (L in total), each level l is formed by n<sub>l</sub> neurons, the i-th neuron (starting from the top, see figure below) of the l-th level (starting from the left, see figure below) is indicated with: **neuron<sub>i</sub><sup>(l)</sup>**.

The set **θ** is a set containing L matrices and L tuples, a matrix and a tuple for each level, with the matrix of the l-th level, indicated with **W<sup>(l)</sup>**, which contains in the i-th row the **weights** of the i-th neuron of the l-th level and the tuple of the l-th level, indicated with **b<sup>(l)</sup>** which contains in the i-th element the **bias** of the i-th neuron of the l-th level.</br>
In reality, even if we have defined θ as a set of matrices and tuples, this is a simplification, the structure of θ is different, in fact this set does not contain the matrices and tuples, but directly all the elements of the matrices and tuples, ordered in a certain way (the first elements are those of W<sup>(1)</sup>, ordered from the first element of the first column of W<sup>(1)</sup>, to the last element of the last column of W<sup>(1)</sup>, then there are the elements of b<sup>(1)</sup> (ordered as they are ordered in b<sup>(1)</sup>), then the elements of W<sup>(2)</sup>, b<sup>(2)</sup>, ..., up to the elements of W<sup>(L)</sup>, b<sup>(L)</sup>) from which matrices and tuples can be easily specified.

The first level is formed by n<sub>1</sub> neurons and each neuron among these, indicated by **neuron<sub>I</sub><sup>(1)</sup>**, has as input the inputs x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n0</sub> of f, as weights the elements of the i-th column of W<sup>(1)</sup> and as bias the i-th element of b<sup>(1)</sup>.

The second level is formed by n<sub>2</sub> neurons and each neuron among them, indicated by **neuron<sub>I</sub><sup>(2)</sup>**, has as input the outputs of all the neurons of the previous level (the first level), as weights the elements of the i-th column of W<sup>(2)</sup> and as bias the i-th element of b<sup>(2)</sup>.

And so on up to the L-th (last) level, formed by n<sub>L</sub> neurons and each neuron among them, indicated by **neuron<sub>I</sub><sup>(L)</sup>**, has as input the outputs of all the neurons of the previous level (the L-1 level), as weights the elements of the i-th column of W<sup>(L)</sup> and as bias the i-th element of b<sup>(L)</sup>, the outputs of the neurons of this last level are the outputs y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>nL</sub> of f, that is, of the neural network itself.

In general, therefore, the network is formed by L levels and each level l is formed by n<sub>l</sub> neurons and each neuron among these, indicated by **neuron<sub>I</sub><sup>(l)</sup>**, has as input the outputs of all the neurons of the previous level (unless l=1, in which case then the inputs are the inputs x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n0</sub> of f), as weights the elements of the i-th column of W<sup>(l)</sup> and as bias the i-th element of b<sup>(l)</sup>, the outputs of the neurons of the last level (l=L) are the outputs y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>nL</sub> of f, that is, of the neural network itself.

#### Definition with matrices:

Given the definition of [product](https://en.wikipedia.org/wiki/Matrix_multiplication) and [sum](https://en.wikipedia.org/wiki/Matrix_addition) between matrices, we can formulate the neural network above also not considering the neuron functions but the level functions, each indicated by **level<sup>(l)</sup>**, which are defined as the "image" in σ of the product between the matrix W<sup>(l)</sup> and the matrix z<sup>(l-1)</sup> (where z<sup>(l-1)</sup> is a matrix formed by a single column (or a vector, or a tuple) containing all the z<sub>i</sub><sup>(l-1)</sup> (ordered according to i from top to bottom)) and with the added matrix b<sup>(l)</sup> (the "image" in σ of the matrix Z resulting from the product and the sum is actually Z but with σ applied to each of its elements, as can be seen from the image below).

The product of the matrix W(l) and the matrix z(l-1) and with the addition of the matrix b(l), is called the preactivation value of level l and is indicated with the symbol a(l) = (a(l)1, ..., a(l)nl), while its image in σ is called the activation value of level l, and is indicated with the symbol z(l) = (z(l)1, ..., z(l)nl).

This representation is the basis of the real representation in a computer of a neural network, since usually dedicated hw and techniques are used to make the product between matrices faster, let's see it:

