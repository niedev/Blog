# How AI Really Works: The Basics in Details

Everyone talks about AI today but [almost no one seems to understand how it works](https://www.reddit.com/r/MachineLearning/comments/r76igz/discussion_rant_most_of_us_just_pretend_to/), this is because there are very few sources that really go into detail and, in any case, almost all are incomplete. After a lot of research and many calculations (too many) I finally managed to write complete notes on the operating principles that underlie modern artificial intelligence.

The field of AI is very broad, technically any program that tries to emulate human intelligence (or a subset of it) is considered artificial intelligence, but what most people mean by AI today are neural networks (more precisely transformers, for modern LLMs).

In this article, I will explain what neural networks are and how their training makes them work. In particular, we will see the simplest neural network, the feed-forward network, which is the basis of how transformers work (maybe an article on how transformers themselves work will come in the future).

This is not a superficial article, as mentioned, I will not spare the details, so, to better understand what I will say, you will need to have a good knowledge of [multivariable calculus](https://en.wikipedia.org/wiki/Multivariable_calculus) and [matrices](https://en.wikipedia.org/wiki/Matrix_(mathematics)).

We are ready to start:

A **Feed Forward Network** (or **multilayer perceptron**) is the simplest type of neural network, that is, a network of artificial neurons, which is inspired by the structure of the neurons in the brain to create, through training, any function that, given certain inputs, produces certain outputs based on the data on which the training was done. 
Although modern neural networks use many optimization techniques for training and more complex neural network structures, the basic principles of their operation are based on the concepts that I will explain in this article (additionally, transformers also have feedforward networks inside them).
Before explaining the structure of a Feed Forward Network, let's see exactly what an artificial neuron is:
</br></br>

## Artificial Neuron

Given a real function of real var **σ**, called the **activation function**, n real values **​​W<sub>1</sub>**, **​​W<sub>2</sub>**, ..., **​​W<sub>n</sub>**, called **weights**, a real value **b** called **bias**:</br>
An **artificial neuron** (or simply **neuron**) is a [scalar function of n real variables](https://en.wikipedia.org/wiki/Function_of_several_real_variables) that receives as input n real values **​​x<sub>1</sub>**, **​​x<sub>2</sub>**, ..., **​​x<sub>n</sub>**, and produces a real value equal to the image, in σ, of the sum of ​​x<sub>1</sub>, ​​x<sub>2</sub>, ..., ​​x<sub>n</sub>, with each x<sub>i</sub> multiplied by the weight W<sub>i</sub>, and with b also added.</br>
Formally, given the premises above, an artificial neuron is the function specified as follows:

**neuron: D<sub>1</sub> x D<sub>2</sub> x ... x D<sub>n</sub> --> R, &nbsp;&nbsp;&nbsp;&nbsp; neuron(x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>)  =  σ(b + W<sub>1</sub> * x<sub>1</sub> + W<sub>2</sub> * x<sub>2</sub> +...+ W<sub>n</sub> * x<sub>n</sub>)**

**N.B.** We'll find out later what exactly weights and biases are for.</br></br>

#### Artificial vs biological neurons:

I'm not a biologist, so take this part with a grain of salt:

Artificial neurons are inspired by biological ones, they are a simpler representation of them, but they should work more or less on the same principles. Although it is still not known exactly how a single biological neuron functions, by combining several artificial neurons it is possible to emulate its behavior (see [this paper](https://www.quantamagazine.org/how-computationally-complex-is-a-single-neuron-20210902/)).
</br></br></br>

## Definition of Feed Forward Network

**A <code>feed forward network</code> is a function that receives as input n<sub>0</sub> real values ​​x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n0</sub>, an activation function σ, and a set θ of matrices and tuples** (containing the weights and biases of each neuron in the network) **and produces as output n<sub>L</sub> real values ​​y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>nL</sub> depending on all inputs. More precisely, the law f is a law defined through the combination of [artificial neurons](https://github.com/niedev/Blog/edit/main/Articles/How_neural_networks_work.md#artificial-neuron) organized in a network-like way, structured in this way:**

![image](https://github.com/user-attachments/assets/a44445fe-f75a-4e29-90ae-696af455ddc7)

<img align="right" width="500" src="https://github.com/user-attachments/assets/672c2d81-e487-4344-85ef-c232b4f19b87">

</br></br>
The network is divided into levels (L in total), each level l is formed by n<sub>l</sub> neurons, the i-th neuron (starting from the top, see figure above) of the l-th level (starting from the left, see figure above) is indicated with: **neuron<sub>i</sub><sup>(l)</sup>**.

The set **θ** is a set containing L matrices and L tuples, a matrix and a tuple for each level, with:
- The matrix of the l-th level, indicated with **W<sup>(l)</sup>**, which contains, in the i-th row, the **weights** of the i-th neuron of the l-th level.
- The tuple of the l-th level, indicated with **b<sup>(l)</sup>** which contains in the i-th element the **bias** of the i-th neuron of the l-th level.</br>

In reality, even if we have defined θ as a set of matrices and tuples, this is a simplification, the structure of θ is different, in fact this set does not contain the matrices and tuples, but directly all the elements of the matrices and tuples, ordered in a certain way (the first elements are those of W<sup>(1)</sup>, ordered from the first element of the first column of W<sup>(1)</sup>, to the last element of the last column of W<sup>(1)</sup>, then there are the elements of b<sup>(1)</sup> (ordered as they are ordered in b<sup>(1)</sup>), then the elements of W<sup>(2)</sup>, b<sup>(2)</sup>, ..., up to the elements of W<sup>(L)</sup>, b<sup>(L)</sup>) from which matrices and tuples can be easily specified.

The first level is formed by n<sub>1</sub> neurons and each neuron among these, indicated by **neuron<sub>i</sub><sup>(1)</sup>**, has:
- As input the inputs x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n0</sub> of f.
- As weights the elements of the i-th column of W<sup>(1)</sup>.
- As bias the i-th element of b<sup>(1)</sup>.

The second level is formed by n<sub>2</sub> neurons and each neuron among them, indicated by **neuron<sub>i</sub><sup>(2)</sup>**, has:
- As input the outputs of all the neurons of the previous level (the first level).
- As weights the elements of the i-th column of W<sup>(2)</sup>.
- As bias the i-th element of b<sup>(2)</sup>.

And so on up to the L-th (last) level, formed by n<sub>L</sub> neurons and each neuron among them, indicated by **neuron<sub>i</sub><sup>(L)</sup>**, has:
- As input the outputs of all the neurons of the previous level (the L-1 level).
- As weights the elements of the i-th column of W<sup>(L)</sup>.
- As bias the i-th element of b<sup>(L)</sup>.

The outputs of the neurons of this last level are the outputs y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>nL</sub> of f, that is, of the neural network itself.

In general, therefore, the network is formed by L levels and each level l is formed by n<sub>l</sub> neurons and each neuron among these, indicated by **neuron<sub>i</sub><sup>(l)</sup>**, has:
- As input the outputs of all the neurons of the previous level (unless l=1, in which case then the inputs are the inputs x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n0</sub> of f)
- As weights the elements of the i-th column of W<sup>(l)</sup>
- As bias the i-th element of b<sup>(l)</sup>.

The outputs of the neurons of the last level (l=L) are the outputs y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>nL</sub> of f, that is, of the neural network itself.</br></br>

#### Definition with matrices:

Given the definition of [product](https://en.wikipedia.org/wiki/Matrix_multiplication) and [sum](https://en.wikipedia.org/wiki/Matrix_addition) between matrices, we can also formulate the neural network above not considering the neuron functions but the level functions, each indicated by **level<sup>(l)</sup>**, which are defined as the "image" in σ of the product between the matrix W<sup>(l)</sup> and the matrix z<sup>(l-1)</sup> (where z<sup>(l-1)</sup> is a matrix formed by a single column (or a vector, or a tuple) containing all the z<sub>i</sub><sup>(l-1)</sup> (ordered according to i from top to bottom)) and with the added matrix b<sup>(l)</sup> (the "image" in σ of the matrix Z resulting from the product and the sum is actually Z but with σ applied to each of its elements, as can be seen from the image below).

The product of the matrix W<sup>(l)</sup> and the matrix z<sup>(l-1)</sup> and with the addition of the matrix b<sup>(l)</sup>, is called the **preactivation value of level l** and is indicated with the symbol **a<sup>(l)</sup> = (a<sup>(l)</sup><sub>1</sub>, ..., a<sup>(l)</sup><sub>nl</sub>)**, while its image in σ is called the **activation value of level l**, and is indicated with the symbol **z<sup>(l)</sup> = (z<sup>(l)</sup><sub>1</sub>, ..., z<sup>(l)</sup><sub>nl</sub>)**.

This representation is the basis of the real representation in a computer of a neural network, since usually dedicated hw and techniques are used to make the product between matrices faster, let's see it:

![image](https://github.com/user-attachments/assets/85fea60d-6d8f-4acb-8e4c-f581dc628b92)

![image](https://github.com/user-attachments/assets/a3d0ab3e-95f6-4353-b9e0-3bea9d2ffc5c)
</br></br>

#### Wait, but why?

Why does a feed-forward neural network have this structure?

Besides emulating the structure of some parts of our brain (yes, not only the artificial neuron is inspired by biology, but also a feed-forward network emulates a biological neural network), the reason behind such an architecture (and most likely also the reason behind the structure of our brain) is that **a feed-forward network can emulate any possible function, with any number of input variables and any number of output variables**. Its accuracy depends on the number of neurons the network has and on the amount and quality of training data, the more they are, the closer the approximated function can be to the real one.

This architecture can basically emulate a function based on the values ​​of its weights and biases, basically, these values ​​determine the shape of the neural network graph (which I remind you is a function), so by modifying these values ​​(through training) we can emulate the graph (and therefore the behavior) of another function of which we only know certain points (training data).

To understand why this is the case and have a visual demonstration I recommend reading [this excellent article by Michael Nielsen](http://neuralnetworksanddeeplearning.com/chap4.html) (this explanation is important, so don't skip it).

This is a stunning result, it basically means that given any problem that can be represented as a function, a neural network can solve it (or that any behavior that can be represented as a function, a neural network can emulate it).

**N.B.** If after reading this you went to buy Open AI shares, chill out, this does not mean that a neural network can solve any problem in practice, but that it can solve any problem that can be represented as a function, for which we have enough data to represent it well and enough parameters to emulate it well. Even if there are many problems (or behaviors) that fall into this category, some will never be able to fall into it and others do not fall into it with current technologies.</br></br></br>


## Training

So far we have analyzed the structure of a neural network, and we have seen that [a neural network can emulate any function](https://github.com/niedev/Blog/edit/main/Articles/How_neural_networks_work.md#wait-but-why), now we will cover how to find the values of the weights and biases that will emulate that function, this process is called training.</br></br>

### Cost function

**Given a feed forward network <code>f(x, σ, θ): D x S<sub>σ</sub> x S<sub>θ</sub> -> C</code>, where D = D<sub>1</sub> x D<sub>2</sub> x...x D<sub>n0</sub>** (x = (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n0</sub>) ∈ D, x<sub>1</sub> ∈ D<sub>1</sub>, x<sub>2</sub> ∈ D<sub>2</sub>, ..., x<sub>n0</sub> ∈ D<sub>n0</sub>)**, C = C<sub>1</sub> x C<sub>2</sub> x...x C<sub>nL</sub>** (y = (y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>nL</sub>) ∈ C, y<sub>1</sub> ∈ C<sub>1</sub>, y<sub>2</sub> ∈ C<sub>2</sub>, ..., y<sub>nL</sub> ∈ C<sub>nL</sub>)**, S<sub>σ</sub> is the set of all possible activation functions σ, S<sub>θ</sub> is the set of all possible combinations of weights and biases θ, and a set T consisting of m pairs of the form (xt ∈ D, yt ∈ C)** (this set represents the **training data of the network**, i.e. a set of correct examples that the network has to emulate, where xt represents the input data of the example and yt the output data that the example produces as a result, usually all the outputs of the examples are produced by humans):

**The <code>cost function of the network f</code> is a [scalar function with k real variables](https://en.wikipedia.org/wiki/Function_of_several_real_variables)** (k is the number of elements of each tuple ∈ S<sub>θ</sub>, i.e. the total number of weights and biases of the neural network f) **of the form:**

**f<sub>C</sub>: S<sub>θ</sub> -> R, &nbsp; with f<sub>C</sub>(θ) = e<sub>fc</sub>(f(xt<sub>1</sub>, σ, θ), yt<sub>1</sub>, f(xt<sub>2</sub>, σ, θ), yt<sub>2</sub>, ..., f(xt<sub>m</sub>, σ, θ), yt<sub>m</sub>)**, &nbsp;&nbsp; where (xt<sub>1</sub>, yt<sub>1</sub>) U (xt<sub>2</sub>, yt<sub>2</sub>) U...U (xt<sub>m</sub>, yt<sub>m</sub>) = T, i.e. (xt<sub>1</sub>, yt<sub>1</sub>), (xt<sub>2</sub>, yt<sub>2</sub>), ..., (xt<sub>m</sub>, yt<sub>m</sub>) are all the elements of T)

**Where e<sub>fc</sub> is an expression that has among the operands f(xt<sub>1</sub>, σ, θ), yt<sub>1</sub>, f(xt<sub>2</sub>, σ, θ), yt<sub>2</sub>, ..., f(xt<sub>m</sub>, σ, θ), yt<sub>m</sub> and represents, for each possible combination of the weights and biases of f, the average amount of variation between the values ​​produced by f with input values ​​from the experiments** (f(xt<sub>i</sub>, σ, θ)) **and the values ​​of the same experiments produced by humans** (yt<sub>i</sub>).

**In other words, f<sub>C</sub> is a function that associates each combination of weights and biases of f with the distance of f from our desired behavior.**

**The f<sub>C</sub> function can have several implementations, but we will only examine one, the <code>quadratic cost function</code>, this is a cost function f<sub>C</sub> defined as:**

**f<sub>C</sub>: S<sub>θ</sub> -> R, &nbsp;&nbsp; with f<sub>C</sub>(θ) = e<sub>fc</sub>(...) = $\frac{(|f(xt_1, σ, θ) - yt_1|)^2 + (|f(xt_2, σ, θ) - yt_2|)^2 +...+ (|f(xt_m, σ, θ) - yt_m|)^2}{m}$**

In practice, this type of cost function associates to each combination of weights and biases of f a value that represents the average squared distance between:
- The results of f with the weights and biases θ (f(xt<sub>i</sub>, σ, θ)).
- The results of the experiments (yt<sub>I</sub>) for each input xt<sub>I</sub> ∈ a pair of T

The bars || indicate the modulus of the result of the subtraction of the two n<sub>L</sub>-tuples, i.e. the distance between the two n<sub>L</sub>-tuples.

**In simple terms, the cost function of a network f is a real function with k real variables** (where k is the total number of weights and biases of the neural network) **that associates to each possible combination of weights and biases of the network f its precision in emulating a behavior that we are interested in** (to calculate this precision we use the training data of the network, that is, a series of tasks that we want the network to be able to perform, performed by humans; more precisely, given a combination of weights and biases, we compare the result of f with that combination and with input the input of each of these training tasks, with the output generated by humans for each of these tasks and we average the results of these comparisons; the technique used for the comparisons depends on the choice of the e<sub>fc</sub> implementation).</br></br>

### Gradient descent

Given the definition of the cost function of a neural network f, if, through [mathematical optimization](https://en.wikipedia.org/wiki/Mathematical_optimization), we find the absolute minimum point of f<sub>C</sub>, then this will be equal to the combination of weights and biases that bring f closer to the desired behavior.

The problem, however, is that this type of calculations on functions that often have thousands (or even billions) of variables (since neural networks often have thousands or billions of neurons) are too complex, even just calculating the weight function for each point is. So, to find the absolute minimum point of a weight function of a neural network, we use a technique that uses calculations that are much lighter in terms of their computation, called gradient descent:

**Gradient descent is based on the assumption that, given a point of the function f<sub>C</sub> of the neural network f** (usually chosen randomly)**, we can compute its [gradient](https://en.wikipedia.org/wiki/Gradient) at that point in a reasonable time** (we will see how to do this later, using the backpropagation technique (link))**. Consequently, given that we know that the gradient of a scalar function with n var. at a point** (in this case ∇f<sub>C</sub>(θ<sub>0</sub>)) **[indicates the direction of maximum growth](https://en.wikipedia.org/wiki/Gradient#Gradient_is_direction_of_steepest_ascent), and that therefore its inverse** (-∇f<sub>C</sub>(θ<sub>0</sub>)) **indicates the direction of maximum decrease, we can find a point of local minimum of f<sub>C</sub>, or a point with a very low value of the image in f<sub>C</sub>, doing the following steps:**

- **Starting from a random point θ<sub>0</sub> of S<sub>θ</sub>** (domain of f<sub>C</sub>)**, we calculate its gradient, we increase θ<sub>0</sub> by adding the opposite of the gradient multiplied by a small value <code>η</code>, called <code>learning rate</code>, in symbols, <code>θ<sub>0</sub> = θ<sub>0</sub> - η*∇f<sub>C</sub>(θ<sub>0</sub>)</code>, that is, we move it in the direction of maximum decrease of f<sub>C</sub>.**
- **We calculate the gradient of the new value of θ<sub>0</sub>, we increase θ<sub>0</sub> again by adding the opposite of the new gradient multiplied by η**.
- **We continue like this until the gradient is (0, 0, ..., 0) or a very close value** (in practice we move the point θ<sub>0</sub> by making it decrease more and more until it coincides with a local minimum point).

<img align=right src="https://github.com/user-attachments/assets/746a64dc-57dd-4594-8d0f-149bcd7d5e8b" width=600 title="Visual example of the path taken by gradient descent in a graph of a cost function, simplified to 3 dimensions (rather than millions or billions)"/>

**N.B.** With this method, we do not calculate the cost function at each point of its domain, nor at θ<sub>0</sub>, we only calculate the gradient of f<sub>C</sub> of the point θ<sub>0</sub> at each movement of θ<sub>0</sub> (which is still a very intensive calculation, in fact usually different optimization techniques are used, including the use of GPUs, stochastic gradient descent, etc).

**N.B.** The learning rate can be a fixed value, but it is usually decreased as the gradient descent algorithm progresses, this allows you to start from a higher value, which allows to skip any local minimum or saddle points encountered at the beginning, when you have not yet descended very far, and gradually decrease it to avoid jumping back and forth to a local minimum point with a very low value (since we have descended for a long time), which would waste a lot of time to get to the end of the training.

At this point, however, we have a problem, in fact, unlike other implementations of machine learning, neural networks almost never produce a [convex](https://en.wikipedia.org/wiki/Convex_function) cost function f<sub>C</sub> (which have only one local minimum point, which is also the global minimum point), so our point θ<sub>0</sub> can stop at a local minimum point or a saddle point with a value much larger than the global minimum point, consequently we almost never use the pure gradient descent method, but a simplified and more imprecise version of it (which is an advantage, as we will see), called stochastic gradient descent. Let's see what it consists of:

**To perform <code>stochastic gradient descent</code> we randomly shuffle the set T containing the training data and divide it into many sets T<sub>1</sub>, T<sub>2</sub>, ..., T<sub>j</sub>, each containing |T|/j elements of T. At this point, we consider for each T<sub>i</sub> the cost function f<sub>i</sub> of f computed only on T<sub>i</sub>, now we perform gradient descent on f<sub>1</sub> but, at each step of the descent, we compute the gradient using a different function** (in order and starting again from the first when we get to the last one) (f<sub>1</sub>, f<sub>2</sub>, ..., f<sub>j</sub>, f<sub>1</sub>, f<sub>2</sub>, ..., f<sub>j</sub>, ...).

<img align=right src="https://github.com/user-attachments/assets/0e0485b8-21af-4f47-9db0-6af761ad395d" width=600 title="Visual example of the path taken by stochastic gradient descent in a graph of a cost function, simplified to 3 dimensions (rather than millions or billions)"/>

This way we can lower the cost of computing the gradient at each step and introduce path imprecision, which will help us avoid getting stuck in local saddle points and minima that are too high.

**N.B.** I remind you that stochastic gradient descent, as well as gradient descent, only calculates the position of the best point θ<sub>0</sub> to descend (in the case of stochastic gradient descent not exactly the best but almost), with the point θ<sub>0</sub> representing a certain combination of all the weights and biases of the neural network, the image of θ<sub>0</sub> in f<sub>C</sub> is calculated only after training (and it is not even mandatory, it is only used to get an idea of ​​the behavior of the neural network trained with the examples provided to train it). Consequently, when we graphically imagine stochastic gradient descent, even if at each step of the descent, the direction is calculated for a different cost function (with therefore a different graph) we can imagine it as a normal gradient descent, therefore on the graph of f<sub>C</sub>, but with the path of θ<sub>0</sub> that, in a small scale, has many more random deviations (in the representation on the buttom, the yellow line always belongs to the point (θ<sub>0</sub>, f<sub>C</sub>(θ<sub>0</sub>))).

A good way to imagine the difference between gradient descent and stochastic gradient descent is to think that the first case is a careful hiker who slowly chooses the steepest possible path to descend, while the second is a drunk who runs and goes a bit here and there, rolls, etc., but in the end arrives at the same destination as the hiker and faster. In fact, if there is a small re-entry before the bottom along the best path the hiker gets stuck in it, while the drunk rolls over it and therefore accidentally overtakes him.

**But why does this method work? Why it manages to stop at a point with an image close to the value of the image of the global minimum point of f<sub>C</sub>** (i.e. close to 0)**, and does not remain stuck at a local minimum point or a saddle point with an image with a value too high? Let's see an informal explanation:**

For the [second gradient](https://www.quora.com/What-is-the-second-derivative-equivalent-of-the-gradient) of f<sub>C</sub> in θ<sub>0</sub> to be [positive definite](https://www.quora.com/What-is-the-second-derivative-equivalent-of-the-gradient) (i.e. θ<sub>0</sub> is a local minimum), all directional derivatives of f<sub>C</sub> in θ<sub>0</sub> must be > 0, and as m increases (number of weights and biases of f, directly proportional to the number of its neurons, which in a deep neural network are usually at least thousands) the number of possible directions (and therefore of directional derivatives) increases exponentially, so when ∇f<sub>C</sub>(θ<sub>0</sub>) is = 0 and m is large it is much more likely that H<sub>fc</sub>(θ<sub>0</sub>) (second gradient of f<sub>C</sub> in θ<sub>0</sub>) is neither > 0 nor < 0 and therefore that θ<sub>0</sub> is a saddle point of f<sub>C</sub> rather than a local minimum point.

But since most critical points are saddle points and local minimum points are rare, it means that in most cases where the function stops there is always a direction in which it can continue to go down, so it is very likely that if we find a local minimum point, it will be close to the value of the global minimum point.

The reason why is that it is more likely to have a local minimum point when the value of the function approaches the global minimum value, since the cost function often has a lower bound of 0 (for example for the quadratic cost function), so when a critical point has an image close to the value 0 it is much more likely that the function will go up in all directions, since it cannot go down for much longer. Another explanation that confirms this, considering the specific implementation of a quadratic cost function, is that, the chances that the values ​​produced by f differ slightly from the results of the experiments is greater than the chance that they are identical, and as the difference between the values ​​of f and the results of the experiments decreases, the chance that weights and biases that generate this difference value exist decreases more and more.

So it is very likely that if we find a local minimum point θ<sub>0</sub>, f<sub>C</sub>(θ<sub>0</sub>) is still close to 0, consequently the points to avoid most with gradient descent are not local minimum points but saddle points, and the stochastic gradient descent method, together with the choice of the right learning rate, introduce imprecision (or noise) and jumps, consequently making it easier to continue descending even in the presence of a saddle point (easier to avoid than a local minimum point), and, even if they are rare, even in the cases of local minimum points that are not too large and steep.

**N.B.** This explanation is informal because (as far as I know, after much research) a formal/mathematical demonstration of why the stochastic gradient works (and why it is more likely that the local minimum points have a low value) is still unknown. Many researches are investigating exactly this and there are many different directions from which this goal can be achieved, mine that I wrote above is only one, quite speculative, more or less like the others.

To increase the efficiency (decrease the probability of getting stuck in a critical point), the speed of stochastic gradient descent and, in general, improve the training and behavior of neural networks, there are several other techniques to apply to gradient descent, for example, running the gradient descent multiple times starting from different points (or deciding on the optimal starting point), optimizations to parallelize it and run it on GPUs, stopping it when a certain value of the cost function is reached, optimizing the learning rate value (also dynamically based on the progress of the descent), changing the type of cost function, etc.</br></br></br>


### Backpropagation

Now the hard part begins. If you want you can consider the backpropagation as a black box that, given the cost function f<sub>C</sub> and a point θ<sub>0</sub> of its domain returns the gradient of f<sub>C</sub> at the point θ<sub>0</sub> (which, as we have seen, will be used by stochastic gradient descent to train the neural network).

But if you want to fully understand how a neural network works, let's see how the backpropagation works:

To help understand the calculations I recommend keeping this scheme open in another window or another device while reading this explanation:

![image](https://github.com/user-attachments/assets/e13d39d3-07c7-4800-ab28-41f9b558b2bb)

**Consider a feed forward network f(x, σ, θ): D x S<sub>σ</sub> x S<sub>θ</sub> -> C, a set T consisting of m pairs of the form (xt ∈ D, yt ∈ C), the set S<sub>θ</sub>, the cost function of f:**

**f<sub>C</sub>: S<sub>θ</sub> -> R, &nbsp;&nbsp; with f<sub>C</sub>(θ) = $\frac{(|f(xt_1, σ, θ) - yt_1|)^2 + (|f(xt_2, σ, θ) - yt_2|)^2 +...+ (|f(xt_m, σ, θ) - yt_m|)^2}{m}$**

**And a point θ<sub>0</sub> ∈ S<sub>θ</sub>, with:**

![image](https://github.com/user-attachments/assets/5f89d263-10ff-4d40-aa55-100898428702)

The variable θ ∈ S<sub>θ</sub> instead is:

![image](https://github.com/user-attachments/assets/6e718cef-a2a5-4ae8-8960-fc9596b92d94)

**N.B.** The point θ<sub>0</sub> and the variable θ are [k-tuples](https://en.wikipedia.org/wiki/Tuple#:~:text=In%20mathematics%2C%20a%20tuple%20is,is%20a%20non%2Dnegative%20integer.) (k is the number of total weights and biases of the neural network f).

**N.B.** The point θ<sub>0</sub> is the point that will be used for the gradient descent, so we start from a random point θ<sub>0</sub>, we calculate the gradient of f<sub>C</sub> in θ<sub>0</sub>, we continue in the direction of the gradient modifying θ<sub>0</sub> accordingly, we recalculate the gradient in the new point θ<sub>0</sub>. We continue in this way until we arrive at a point θ<sub>0</sub> that is a local minimum (or global, but it is almost impossible) such that f<sub>C</sub>(θ<sub>0</sub>) is low enough.

I note that S<sub>θ</sub> can also be written as: S<sub>θ1</sub> x S<sub>θ2</sub> x ... x S<sub>θk</sub> where S<sub>θd</sub> (with 1 <= d <= k) is the membership set of the d-th component of θ.

Now to simplify the calculations let's consider a series of cost functions, one for each element of T, more precisely let's consider m cost functions of the form:

**f<sub>Cb</sub>: S<sub>θ</sub> -> R,  &nbsp;&nbsp; with f<sub>Cb</sub>(θ) = $\frac{(|f(xt_b, σ, θ) - yt_b|)^2}{1}$ = $(|f(xt_b, σ, θ) - yt_b|)^2$**, &nbsp;&nbsp; 1 <= b <= m

**N.B.** b<sup>(l)</sup> is the bias vector of level l, while b is just a natural number between 1 and m, the two symbols have nothing to do with each other.

For the [definition of gradient](https://en.wikipedia.org/wiki/Gradient) and the properties of gradient:

<img align=left src="https://github.com/user-attachments/assets/ac8b222d-3075-446a-9d6d-9e66bd7e23cf" width=600/>
<BR CLEAR="all"><br/>

**Now we need to understand how to calculate all the partial derivatives of f<sub>Cb</sub> in θ<sub>0</sub>** (that is, the partial derivatives in θ<sub>0</sub> with respect to each component of θ)**, to do this we need to perform the following steps:**

1) **We will formulate f<sub>Cb</sub> as a combination of functions, then we will use the chain rule to find a formula that allows us to calculate** (via the chain rule) **the gradient of f<sub>Cb</sub> with respect to the preactivation values (link) ​​of any level l** (which we will denote with δ<sup>(l)</sup>).

2) **After that, we will use the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) to calculate the gradient of any level l with respect to W<sup>(l)</sup> at the point W<sub>0</sub><sup>(l)</sup> considering the combination between the function f<sub>Cb</sub>, which has as independent variables the preactivation values ​​of level l** (which therefore has as gradient δ<sup>(l)</sup>)**, and the preactivation function of level l, which has as independent variables the components of W<sup>(l)</sup>** (which we indicate with a<sup>(l)</sup>(W<sup>(l)</sup><sub>1,1</sub>, ..., W<sup>(l)</sup><sub>nl,nl-1</sub>))**, which therefore, by the chain rule, will be equal to δ<sup>(l)</sup> multiplied by the gradient of a<sup>(l)</sup>(W<sup>(l)</sup><sub>1,1</sub>, ..., W<sup>(l)</sup><sub>nl,nl-1</sub>) at the point W<sub>0</sub><sup>(l)</sup>** (which, as we will see, is equal to δ<sup>(l)</sup> ⨂ z<sup>(l−1)</sup><sub>0</sub>)**, finally we will see that this gradient will have, as components, values ​​equal to the partial derivatives with respect to each weight of the level l: (f<sub>Cb</sub>)'<sub>W<sup>(l)</sup><sub>i,j</sub></sub>(θ<sub>0</sub>).**

3) **Finally we will use the same principles of point 2) to calculate the partial derivatives with respect to each bias of the level l: (f<sub>Cb</sub>)'<sub>b<sup>(l)</sup><sub>i</sub></sub>(θ<sub>0</sub>).**

**After that, thanks to what we have said here, we will have found formulas to calculate the gradient of f<sub>Cb</sub> at the point θ<sub>0</sub>, and therefore to calculate also the gradient of f<sub>C</sub> at the point θ<sub>0</sub>, so thanks to these formulas we will have all the bases to formulate the backpropagation algorithm to find the gradient of f<sub>C</sub> at the point θ<sub>0</sub> with the minimum number of calculations possible.**

**N.B.** <ins>If you have not understood everything from this explanation proceed step by step from here to the end and things will be clearer.</ins> </br></br>

#### Point 1:

**First, let's decide how to formulate f<sub>Cb</sub> as a combination, so that we can apply the chain rule and find the formula for δ<sup>(l)</sup>:**

For what we have said above, f<sub>Cb</sub>(θ) = $(|f(xt_b, σ, θ) - yt_b|)^2$, where yt<sub>b</sub> is a constant, and $f(xt_b, σ, θ)$, by its definition, is a function composed of L level functions, with each level l function being composed of an activation function (which has as input the preactivation values ​​of this level a<sup>(l)</sup> = (a<sup>(l)</sup><sub>1</sub>, ..., a<sup>(l)</sup><sub>nl</sub>) and as output the activation values ​​z<sup>(l)</sup> = (z<sup>(l)</sup><sub>1</sub>, ..., z<sup>(l)</sup><sub>nl</sub>)) and a preactivation function (which has as input the activation values ​​of the previous level z<sup>(l-1)</sup> = (z<sup>(l-1)</sup><sub>1</sub>, ..., z<sup>(l-1)</sup><sub>nl-1</sub>) and as output the preactivation values ​​a<sup>(l)</sup> = (a<sup>(l)</sup><sub>1</sub>, ..., a<sup>(l)</sup><sub>nl</sub>)) (normally these functions would also have as input the activation function σ and the weights and biases θ, but in this case we will consider them as constants). Consequently we can consider f<sub>Cb</sub> as a function that has as independent variable the inputs x = (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n0</sub>) of the neural network f (instead of the variable θ that contains the weights and biases of the network, which instead in our representation will be considered as a constant θ<sub>0</sub>) and that is the composition between f<sub>Cb</sub> and the compositions that form the function $f(xt_b, σ, θ)$ that we have described above, formally:

<img align=left src="https://github.com/user-attachments/assets/1885c809-881c-47b9-829f-953427add90d" width=1100>
<BR CLEAR="all"><br/>

Recall that a matrix of the form W<sub>0</sub><sup>(l)</sup> is the (constant) matrix containing the weights of level l contained in the combination of weights and biases θ<sub>0</sub> (i.e. the point whose gradient we want to calculate) and a tuple of the form b<sub>0</sub><sup>(l)</sup> is the (constant) tuple containing the biases of level l contained in the combination of weights and biases θ<sub>0</sub>.

**N.B.** I anticipate that at the beginning of the algorithm, a feedforward will be performed, that is, we will execute the neural network f with input xt<sub>b</sub> = (xt<sub>b1</sub>, xt<sub>b2</sub>, ..., xt<sub>bn0</sub>) and with parameters θ<sub>0</sub> and we will save at each level l all the preactivation (a<sup>(l)</sup><sub>0</sub> = (a<sup>(l)</sup><sub>01</sub>, ..., a<sup>(l)</sup><sub>0nl</sub>)) and activation (z<sup>(l)</sup><sub>0</sub> = (z<sup>(l)</sup><sub>01</sub>, ..., z<sup>(l)</sup><sub>0nl</sub>)) values, so we already know their (constant) values ​​for f with input xt<sub>b</sub> = (xt<sub>b1</sub>, xt<sub>b2</sub>, ..., xt<sub>bn0</sub>) and with parameters θ<sub>0</sub> (if we consider our combination, instead of f, then the a<sup>(l)</sup><sub>0</sub> and the z<sup>(l)</sup><sub>0</sub> are the values ​​that we encounter if we execute our combination with input xt<sub>b</sub> = (xt<sub>b1</sub>, xt<sub>b2</sub>, ..., xt<sub>bn0</sub>) (since it has already the parameters fixed on θ<sub>0</sub>)).<br/><br/>


Let's look at these functions in more detail, examining their component functions (remember that z<sup>(L)</sup> and a<sup>(L)</sup> are vector functions with multiple real variables and are therefore specified using component functions, while f<sub>Cb</sub> is a scalar function with multiple real variables and is therefore specified normally):

<img align=left src="https://github.com/user-attachments/assets/20611455-5e1d-4aaa-a516-ac4ba748c652" width=1100>
<BR CLEAR="all"><br/>

**N.B.** I remind you that a matrix of the form **W<sub>0</sub><sup>(l)</sup>** is the (constant) matrix containing the weights of level l contained in the combination of weights and biases θ<sub>0</sub> (i.e. the point whose gradient we want to calculate) and a tuple of the form b<sub>0</sub><sup>(l)</sup> is the (constant) tuple containing the biases of level l contained in the combination of weights and biases θ<sub>0</sub>.<br/><br/>


Given the property of combination of vector functions with real multiple vars, considering two vector functions f and g with real multiple vars:

$∇(f(g(x_{01}, …,x_{0n})))= ∇(f)(g(x_{01}, …,x_{0n})) ∗ ∇(g(x_{01}, …,x_{0n}))$

Where $∇(f)(g(x_1, …,x_n))$ is equal to the gradient of $∇(f(t_1, …,t_k))$ with, once calculated, in place of t<sub>1</sub>, …, t<sub>k</sub> the component functions of g, i.e. g<sub>1</sub>(x<sub>01</sub>, …, x<sub>0n</sub>), …, g<sub>k</sub>(x<sub>01</sub>, …, x<sub>0n</sub>), in other words:

$∇(f)(g(x_{01}, …, x_{0n})) = ∇(f)(g_1(x_{01}, …, x_{0n}),…, g_k (x_{01}, …, x_{0n})) =$ <br/>
$= (f_{g_1(x_1,…,x_k)}′(g_1(x_{01}, …, x_{0n}),…, g_k(x_{01}, …, x_{0n})), …, f_{g_k (x_1,…,x_k)}′(g_1(x_{01}, …, x_{0n}),…, g_k(x_{01}, …, x_{0n})))$

That is, the functions g<sub>1</sub>(x<sub>1</sub>, …, x<sub>n</sub>), …, g<sub>k</sub>(x<sub>1</sub>, …, x<sub>n</sub>) are considered as the independent variables of f &nbsp; (its graph therefore instead of the axes x<sub>1</sub>, ..., x<sub>n</sub> has axes g<sub>1</sub>(x<sub>1</sub>, …, x<sub>n</sub>), …, g<sub>k</sub>(x<sub>1</sub>, …, x<sub>n</sub>)).<br/><br/>


So, if we consider **a<sup>(1)</sup>** (= a<sup>(1)</sup>(x<sub>1</sub>, ..., x<sub>n0</sub>) = (a<sup>(1)</sup><sub>1</sub>, ..., a<sup>(1)</sup><sub>n1</sub>) = (a<sup>(1)</sup><sub>1</sub>(x<sub>1</sub>, ..., x<sub>n0</sub>), ..., a<sup>(1)</sup><sub>n1</sub>(x<sub>1</sub>, ..., x<sub>n0</sub>))) as the independent variable of **f<sub>Cb</sub>** and the point **a<sup>(1)</sup><sub>0</sub>** which is the image of a<sup>(1)</sup> for x = x<sub>tb</sub> (and θ = θ<sub>0</sub>) (obtained via the initial feedforward), we get that the gradient of $f_{Cb}(z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(1)} (a^{(1)})))))))$ at the point a<sup>(1)</sup><sub>0</sub> is:

<img align=left src="https://github.com/user-attachments/assets/da6bbce3-1962-4c4f-ac73-004c2e6e62c2" width=1200>
<BR CLEAR="all"><br/><br/>


**Now let's see what each gradient is equal to** (we calculate the gradient of the individual functions of the composition, not of the entire composition, therefore considering the variables internal to the composition as an independent variable, at the notational level the difference is that in the case of the gradient of the entire composition the gradient symbol includes the entire composition, while, in the case of the gradient of only one of the functions of the composition, the gradient symbol includes only the law of the function whose gradient we want to calculate)**:**

- According to the definition of gradient of a scalar function with multiple real variables at a point (the point is a<sup>(1)</sup><sub>0</sub>):

  <img align=left src="https://github.com/user-attachments/assets/cacd9d6f-062d-499d-860f-d7f120c4887a" width=1100>
  <BR CLEAR="all">
  
  Instead of ... in the last expression we have $`a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(1)} (a^{(1)}_0)))))`$, given the definition of $`a^{(1)}_{0}, a^{(1)}_{0} = a^{(1)} (xt_{b1}, …,xt_{bn_0})`$, then, for any i between 1 and nL: $`z^{(L)}_i (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(1)} (a^{(1)}_0))))))`$ = $`z^{(L)}_i (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(1)} (a^{(1)} (xt_{b1}, …,xt_{bn_0})))))))`$, which is the image of $`z^{(L)}_i`$ for x = xt<sub>b</sub> (and θ = θ<sub>0</sub>) (constant), which we denote with $`z^{(L)}_{0i}`$ (which coincides with the notation we used for $`a^{(1)}_0`$ in fact it is one of the values ​​we save with the initial feedforward), so we substitute it in the last expression, obtaining:
  
  <img align=left src="https://github.com/user-attachments/assets/bd7146c7-eebc-49fa-9d82-a1ae594e56dc" width=450>
  <BR CLEAR="all">

  Where the expressions of the partial derivative functions are already known a priori (they only depend on the type of cost function), then the computer only has to calculate their images for the value z<sup>(L)</sup><sub>0</sub> = (z<sup>(L)</sup><sub>01</sub>, ..., z<sup>(L)</sup><sub>0nL</sub>) that it saved in the first forward pass.

- The gradient of any z<sup>(l)</sup> (i.e. of the z of any level l) at the point a<sup>(1)</sup><sub>0</sub> is a matrix that has each row containing the gradient in a<sup>(1)</sup><sub>0</sub> of one of the component functions of z<sup>(l)</sup>, i.e. all the images in a<sup>(1)</sup><sub>0</sub> of the partial derivative functions of one of the component functions of z<sup>(l)</sup> (which are scalar functions with multiple real vars), but the component functions of z<sup>(l)</sup> (the functions that constitute the rows of the matrix in the second passage of the equation below) even if mathematically have as independent variables all the variables a<sup>(l)</sup><sub>1</sub>, ..., a<sup>(l)</sup><sub>nl</sub>, each in fact depends only on one of these variables, so it is as if it had in its law only one of these variables, consequently, their partial derivative function with respect to the variables not contained in the law is 0 (so is its image in a<sup>(1)</sup><sub>0</sub>, i.e. the partial derivative in a<sup>(1)</sup><sub>0</sub>) and the partial derivative function with respect to the variable they contain is equal to D(σ(x)) = σ′(x)∗D(x) = σ′(x)∗1 = σ′(x) (where x is the independent variable = a<sup>(l)</sup><sub>I</sub>, with i being the index of the variable on which the component function of z<sup>(l)</sup> considered depends), then its image in a<sup>(1)</sup><sub>0</sub>, i.e. the partial derivative in a<sup>(1)</sup><sub>0</sub> is equal to $`σ′(a^{(l)}_i (z^{(l−1)} (a^{(l−1)} (…(z^{(1)} (a^{(1)}_0))))))`$ which by the same principles used here is equal to $`σ′(a^{(l)}_{0i})`$, where $`a^{(l)}_{0i}`$ is the image of $`z^{(L)}_i`$ for x = xt<sub>b</sub> (and θ = θ<sub>0</sub>) (constant), whose notation coincides with the notation we used for a<sup>(1)</sup><sub>0</sub> (in fact it is one of the values ​​we save with the initial feedforward), so the calculation of the gradient of z<sup>(l)</sup> in a<sup>(1)</sup><sub>0</sub> is the following:

  <img align=left src="https://github.com/user-attachments/assets/9fd7176c-eb07-4196-8f01-895a4d3d52d5" width=800>
  <BR CLEAR="all">

  Here too the derivative of σ is known a priori, so the computer, to obtain the last matrix of the above equality, only has to calculate its image for the constant values ​​a<sup>(l)</sup><sub>01</sub>, ..., a<sup>(l)</sup><sub>0nl</sub> that it saved in the first forward pass.

- We can calculate the gradient of any a<sup>(l)</sup> (i.e. of the a of any level l) at the point a<sup>(1)</sup><sub>0</sub> by applying the same principles as the result above:

  <img align=left src="https://github.com/user-attachments/assets/5232226d-0900-460a-a10c-42a73da863ea" width=1100>
  <BR CLEAR="all">

  This matrix is ​​known a priori, since its values ​​are components of θ<sub>0</sub> (which is already known, being the point whose gradient we want to calculate).

  **N.B.** I remind you that a matrix of the form W<sub>0</sub><sup>(l)</sup> is the (constant) matrix containing the weights of level l contained in the combination of weights and biases of θ<sub>0</sub> (i.e. the point whose gradient we want to calculate) and a tuple of the form b<sub>0</sub><sup>(l)</sup> is the (constant) tuple containing the biases of level l contained in the combination of weights and biases of θ<sub>0</sub> (i.e. the point whose gradient we want to calculate). <br/><br/>

**So, substituting the gradients according to the formulas we found, we can rewrite the above equation as:**

<img align=left src="https://github.com/user-attachments/assets/59a4af17-a0ba-4d75-b4b0-990dee668b83" width=1200>
<BR CLEAR="all"><br/>

The first multiplication produces a vector of length nL, the second a vector of length nL-1, the third a vector of length nL-1, the fourth a vector of length nL-2, ..., the penultimate a vector of length n1, the last a vector of length n0.

It is also important to know that the expression above is made up only of constants, so it produces a constant.

**N.B.** I remind you that a matrix of the form W<sub>0</sub><sup>(l)</sup> is the (constant) matrix containing the weights of level l contained in the combination of weights and biases of θ<sub>0</sub> (i.e. the point whose gradient we want to calculate) and a tuple of the form b<sub>0</sub><sup>(l)</sup> is the (constant) tuple containing the biases of level l contained in the combination of weights and biases of θ<sup>0</sup> (i.e. the point whose gradient we want to calculate). <br/><br/>


Since $`∇(z^{(l)})(a^{(l)} (z^{(l−1)} (a^{(l−1)} (…(z^{(1)} (a^{(1)}_0))))))`$ is always a square diagonal matrix of size nl x nl, and, analyzing what we wrote above, it is always multiplied by a vector (or tuple) of length nl, by this property, we can rewrite it as a vector and always multiply it by the [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)), so in general, we can rewrite the above equation as:

<img align=left src="https://github.com/user-attachments/assets/ddf7669f-137e-473b-a4ff-c7162d1e8971" width=1200>
<BR CLEAR="all"><br/>

**N.B.** This rewriting is useful because it saves space and time for the computer during the execution of the algorithm. <br/><br/>


**Based on this we can then write the formula to calculate the constants δ<sup>(l)</sup>:**

<img align=left src="https://github.com/user-attachments/assets/96258d9c-9124-47e1-bed4-8dc84b9fc54f" width=1200>
<BR CLEAR="all"><br/>

Recall that **δ<sup>(l)</sup>** is a constant vector (or tuple) of length nl, which we denote with: **(δ<sup>(l)</sup><sub>1</sub>, δ<sup>(l)</sup><sub>2</sub>, …, δ<sup>(l)</sup><sub>nl</sub>)**<br/><br/>

#### Point 2:

Now, let's consider the function $`f_{Cb}(z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)} (W^{(l)})))))))))`$, that is, the function equal to the one defined at the beginning of point 1 (link) but that instead of the function z<sup>(l+1)</sup> (which depends on the inputs x = (x<sub>1</sub>, ..., x<sub>n0</sub>)) has the independent variable **W<sup>(l)</sup>**, more precisely the function a<sup>(l)</sup> has instead of the constant W<sub>0</sub><sup>(l)</sup> the independent variable W<sup>(l)</sup> and instead of the variable $`z^{(l−1)} (a^{(L−1)} (…(z^{(1)}(a^{(1)} (x_1, …, x_{n_0} )))))`$ has the constant z<sup>(l-1)</sup><sub>0</sub> obtained via the initial feedforward (b<sub>0</sub><sup>(l)</sup> remains the same), in symbols:   
$`a^{(l)} (W^{(l)})= W^{(l)} ∗ z^{(l−1)}_0 + b_0^{(l)}`$

Using the formula to calculate δ<sup>(l)</sup> we can prove the following equation, which finds **the gradient of the function $`f_{Cb}(z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)} (W^{(l)})))))))))`$**, which we defined above, **at the point W<sub>0</sub><sup>(l)</sup>** (which will bring us to the final equation of point 2, to find $`(f_{Cb})_{W^{(l)}_{(i,j)}}(θ_0)`$ knowing only the value of $`δ^{(l)}_i`$ and $`z^{(l−1)}_{0j}`$):

<img align=left src="https://github.com/user-attachments/assets/ec2a6ea8-766d-40d5-8b4a-b7c16bb78529" width=1200>
<BR CLEAR="all"><br/>

Where I remind you that **z<sup>(l−1)</sup><sub>0</sub>** is one of the values ​​we find during the initial feedforward, which more precisely is the image of z<sup>(1−1)</sup> for x = xt<sub>b</sub> (and θ = θ<sub>0</sub>). And where ⨂ is the [tensor product](https://www.math3ma.com/blog/the-tensor-product-demystified) between two tuples, that is, the product that, given two tuples, produces a tuple such that each of its elements is the product of an element of the first tuple and an element of the second tuple, and so on for all combinations of 2 elements, one of the first tuple and one of the second.

**N.B.** I also remind you that a matrix of the form W<sub>0</sub><sup>(l)</sup> is the (constant) matrix containing the weights of level l contained in the combination of weights and biases of θ<sub>0</sub> (i.e. the point for which we want to calculate the gradient) and a tuple of the form b<sub>0</sub><sup>(l)</sup> is the (constant) tuple containing the biases of level l contained in the combination of weights and biases of θ<sub>0</sub> (i.e. the point for which we want to calculate the gradient).

___

#### Proof of the above equation:

The first step of this equation in practice is the application of the chain rule, since we consider $`f_{Cb}(z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)} (W^{(l)}))))))))`$ as the combination of $`f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)}(t)))))))`$ and $`a^{(l)} (W^{(l)})`$, with therefore $`t = a^{(l)} (W^{(l)})`$.

So, since we have proved the first step, now we only need to prove the second step, to do that we need to prove that $`a^{(l)}_0 = a^{(l)} (W_0^{(l)})`$ and that $`∇(f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)}_0)))))))) ∗ ∇a^{(l)} (W_0^{(l)}) = δ^{(l)} ⨂ z^{(l−1)}_0`$:

We first prove that $`a^{(l)}_0 = a^{(l)} (W_0^{(l)})`$: By the definition of $`a^{(l)} (W^{(l)})`$, the latter is a function $`a^{(l)} (W^{(l)}) = W^{(l)}∗z^{(l−1)}_0 + b_0^{(l)}`$, but its image for $`W_0^{(l)}`$ transforms the variable W<sup>(l)</sup> into the constant W<sub>0</sub><sup>(l)</sup>, so $`a^{(l)} (W_0^{(l)}) = W_0^{(l)} ∗ z^{(l−1)}_0 + b_0^{(l)}`$, this image however coincides with the definition of a<sup>(l)</sup><sub>0</sub> (saved in the initial feedforward), since this constant is defined as the preactivation value of level l (a<sup>(l)</sup><sub>0</sub> = (a<sup>(l)</sup><sub>01</sub>, ..., a<sup>(l)</sup><sub>0nl</sub>)) of the combination defined at the beginning of point 1 (link) with input xt<sub>b</sub> = (xt<sub>b1</sub>, xt<sub>b2</sub>, ..., xt<sub>bn0</sub>) (it already has the parameters fixed on θ<sub>0</sub>), that is, given the definition of a<sup>(l)</sup>, $`a^{(l)}_0 = W_0^{(l)} ∗ z^{(l−1)} (a^{(L−1)} (…(z^{(1)} (a^{(1)} (xt_{b1}, …, xt_{bn_0}))))) + b_0^{(l)}`$, and since, by the definition of z<sup>(l-1)</sup><sub>0</sub> (also one of the values ​​saved in the initial feedforward), $`z^{(l-1)}_0 = z^{(l−1)} (a^{(L−1)} (…(z^{(1)} (a^{(1)} (xt_{b1}, …, xt_{bn_0})))))`$, then $`a^{(l)}_0 = W_0^{(l)} ∗ z^{(l−1)} (a^{(L−1)} (…(z^{(1)} (a^{(1)} (xt_{b1}, …, xt_{bn_0}))))) + b_0^{(l)} = W_0^{(l)} ∗ z^{(l−1)}_0 + b_0^{(l)}`$, so $`a^{(l)}_0 = a^{(l)} (W_0^{(l)})`$.

Now instead we prove that $`∇(f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)}_0)))))))) ∗ ∇a^{(l)} (W_0^{(l)}) = δ^{(l)} ⨂ z^{(l−1)}_0`$: First, by the definition of a<sup>(l)</sup>(W<sup>(l)</sup>):

<img align=left src="https://github.com/user-attachments/assets/92011f1c-cc9c-4714-9064-2a8c64b50d82" width=1200>
<BR CLEAR="all"><br/>

I remind you that the $`z^{(l−1)}_{0i}`$ and the $`{b_0}^{(l)}_{j}`$ are constants, while the $`W^{(l)}_{i,j}`$ are the independent variables, moreover, for the same concepts seen in the second definition of the gradients found in point 1:

<img align=left src="https://github.com/user-attachments/assets/b2909672-4e81-47e8-8a43-33ec3e545221" width=750>
<BR CLEAR="all"><br/>

The matrix produced is a band matrix of height nl.

At this point given that by the definition of δ<sup>(l)</sup>: $`∇(f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)}_0))))))))`$ = $`δ^{(l)}`$ = $`(δ^{(l)}_1, δ^{(l)}_2,…, δ^{(l)}_{nl})`$ = $`[δ^{(l)}_1 \, δ^{(l)}_2 … δ^{(l)}_{nl}]`$ (1 x nl matrix), considering the definition of [scalar product](https://en.wikipedia.org/wiki/Dot_product), [matrix product](https://en.wikipedia.org/wiki/Matrix_multiplication#:~:text=For%20matrix%20multiplication%2C%20the%20number,B%20is%20denoted%20as%20AB.) and [tensor product](https://www.math3ma.com/blog/the-tensor-product-demystified) (⨂):

<img align=left src="https://github.com/user-attachments/assets/2d2b19a5-06c9-4f0a-91c5-c400fda5b7fe" width=900>
<BR CLEAR="all"><br/>

So we have proved that: $`∇(f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)} ({W_0}^{(l)}))))))))`$ = $`δ^{(l)} ⨂ z^{(l−1)}_0`$ , so we have proved the above equation.

___

Now, using the above equation, we derive a formula for the partial derivatives of f<sub>Cb</sub> with respect to each weight of a level l, i.e. the final formula of this point, to do so we first consider the definition of the gradient as a tuple of partial derivatives and the above equation, we also denote (to simplify the notation) the function $`f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)} ({W_0}^{(l)}))))))))`$ with the symbol $`g(W_0^{(l)})`$:

<img align=left src="https://github.com/user-attachments/assets/1fb74db4-1e3c-4d73-a91e-6c67bc1dfcbd" width=900>
[EQ. 2-1]
<BR CLEAR="all"><br/>

Now let's consider the initial function f<sub>Cb</sub>, that is the function f<sub>Cb</sub>: S<sub>θ</sub> -> R, with $`f_{Cb}(θ) = (f(xt_b, σ, θ) − yt_b)^2`$, where therefore the activation function σ and the input xt<sub>b</sub> are constants, while all the weights and biases are variables (θ), this function can also be written through our initial composite function modified so that it has as independent variable θ and instead of the variable x the constant xt<sub>b</sub>, that is as the function $`f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(1)} (a^{(1)}(θ)))))))`$.

Now let us consider a partial derivative of f<sub>Cb</sub>(θ) at the point θ<sub>0</sub> of the form $`{(f_{Cb})'}_{{W}^{(l)}_{i,j}}(θ_0) \, (= {(f_{Cb})'}_{{W}^{(l)}_{i,j}} (W^{(1)}_{1,1}, ..., W^{(l)}_{i,j}, ..., {b_0}^{(L)}_{nL}))`$, the latter, for what we have said above and for the definition of partial derivative at a point, is a value that represents the slope in the direction $`W^{(l)}_{i,j}`$ of the function $`f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(1)} (a^{(1)} (θ)))))))`$ at the point θ<sub>0</sub>, consequently the components other than $`W^{(l)}_{i,j}`$ are irrelevant, and can assume their constant value that they have at the point θ<sub>0</sub> (very informal reasoning, but the result is correct), therefore we can intuit that this value is the same if we consider the partial derivative of direction $`W^{(l)}_{i,j}`$ of the same function, but which has as independent variables only the components of θ that belong to W<sup>(l)</sup> and has the other components of θ equal to their constant versions (∈ θ<sub>0</sub>), at the point $`{W_0}^{(l)}`$ (which would be the constant version (∈ θ<sub>0</sub>) of the components of the independent var. W<sup>(l)</sup>), that is, that value is the same as the partial derivative with respect to $`W^{(l)}_{i,j}`$ of $`∇(f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(1)} (a^{(1)} (W^{(l)})))))))))`$ at the point $`{W_0}^{(l)}`$, but since, due to the way the neural network f is structured, the change of $`W^{(l)}_{i,j}`$ only affects the levels from l to L, then the partial derivative with respect to $`W^{(l)}_{i,j}`$ of $`∇(f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(1)} (a^{(1)} (W^{(l)})))))))))`$ at the point $`{W_0}^{(l)}`$ will be equal to the partial derivative with respect to $`W^{(l)}_{i,j}`$ of $`∇(f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)} (W^{(l)})))))))))`$ at the point $`{W_0}^{(l)}`$, with $`a^{(l)} (W^{(l)}) = W^{(l)} ∗ z^{(l−1)}_0 + {b_0}^{(l)}`$, where the constant z<sup>(l-1)</sup><sub>0</sub> is the one obtained through the initial feedforward (i.e. the image of z<sup>(l−1)</sup> (…) for x = xt<sub>b</sub> (and θ = θ<sub>0</sub>)), then given that the function just described is equal to g(W<sup>(l)</sup>), consequently $`{(f_{Cb})'}_{{W}^{(l)}_{i,j}}(θ_0) = {g'}_{{W}^{(l)}_{i,j}}({W}^{(l)}_0)`$, then:

<img align=left src="https://github.com/user-attachments/assets/3173840a-059b-4633-9015-f79a0b44cfe1" width=580>
[EQ. 2-2]
<BR CLEAR="all"><br/>

**N.B.** We must not confuse the initial f<sub>Cb</sub> with that of the composition, in fact, the initial f<sub>Cb</sub> includes the entire composition (but with independent variable θ instead of x), while the f<sub>Cb</sub> of the composition is only the most external function of the composition, even if they have the same name as the law they are two different functions, with different laws and independent variables.

**N.B.** The modification of the parameter $`W^{(l)}_{i,j}`$ does not modify in any way the other parameters of the neural network, these values ​​are always independent of each other, it only affects the final value of all the levels between l and L.

Given the [EQ. 2-1] and [EQ. 2-2], we can substitute the tuple $`((f_{Cb})′_{W^{(l)}_{1,1}} (θ_0), (f_{Cb})′_{W^{(l)}_{1,2}} (θ_0), …, (f_{Cb})′_{W^{(l)}_{nl,nl−1}} (θ_0))`$ for the third term of this equation, obtaining (considering only some of the terms of [EQ. 2-1]):

<img align=left src="https://github.com/user-attachments/assets/2e36d94c-e823-4fc8-adc1-8162e26dd024" width=950>
<BR CLEAR="all"><br/>

So, by the definition of equality between tuples, since the first and third terms are equal, their respective components are also equal, consequently we can say that:

<img align=left src="https://github.com/user-attachments/assets/076ae4ac-d45b-44a2-89a9-146c35154edd" width=250>
<BR CLEAR="all"><br/>

#### Point 3:

In point 3) we will use the same principles as in point 2) to compute the partial derivatives with respect to each bias (instead of each weight) of level l: (f<sub>Cb</sub>)'<sub>b<sup>(l)</sup><sub>i</sub></sub>(θ<sub>0</sub>).

I remind you of the definition of **δ<sup>(l)</sup>**:

<img align=left src="https://github.com/user-attachments/assets/f0d11d0f-f3ad-45f4-9f15-476ca0442358" width=1200>
<BR CLEAR="all"><br/>

Recall that **δ<sup>(l)</sup>** is a constant vector (or tuple) of length nl, which we denote with: **(δ<sup>(l)</sup><sub>1</sub>, δ<sup>(l)</sup><sub>2</sub>, …, δ<sup>(l)</sup><sub>nl</sub>)**<br/><br/>


Now, let's consider the function $`f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)} (b^{(l)})))))))))`$, that is, the function equal to the one defined at the beginning of point 1 (link) but that instead of the function z<sup>(l+1)</sup> (which depends on the inputs x = (x<sub>1</sub>, ..., x<sub>n0</sub>)) has the independent variable b<sup>(l)</sup>, more precisely the function a<sup>(l)</sup> has, instead of the constant b<sub>0</sub><sup>(l)</sup>, the independent variable b<sup>(l)</sup> and, instead of the variable $`z^{(l−1)} (a^{(L−1)} (…(z^{(1)} (a^{(1)} (x_1, …, x_{n_0})))))`$, has the constant z<sup>(l-1)</sup><sub>0</sub> obtained by the initial feedforward (W<sub>0</sub><sup>(l)</sup> remains the same), in symbols: <br/>
$`a^{(l)} (b^{(l)}) = W_0^{(l)} ∗ z^{(l−1)}_0 + b^{(l)}`$

Using the formula to calculate δ<sup>(l)</sup> we can prove the following equation, which finds **the gradient of the function $`f_{Cb}(z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)} (b^{(l)})))))))))`$**, which we defined above, **at the point b<sub>0</sub><sup>(l)</sup>** (which will lead us to the final equation of point 3, to find $`(f_{Cb})_{b^{(l)}_i} (θ_0)`$ knowing only $`δ^{(l)}_i`$):

<img align=left src="https://github.com/user-attachments/assets/d2f19b1f-a3df-49d5-bcd4-a5a4da66d6e2" width=1200>
<BR CLEAR="all"><br/>

Where I remind you that **z<sup>(l−1)</sup><sub>0</sub>** is one of the values ​​we find during the initial feedforward, which more precisely is the image of z<sup>(l−1)</sup> for x = xt<sub>b</sub> (and θ = θ<sub>0</sub>).

**N.B.** I also remind you that a matrix of the form **W<sub>0</sub><sup>(l)</sup>** is the (constant) matrix containing the weights of level l contained in the combination of weights and biases of θ<sub>0</sub> (i.e. the point for which we want to calculate the gradient) and a tuple of the form **b<sub>0</sub><sup>(l)</sup>** is the (constant) tuple containing the biases of level l contained in the combination of weights and biases of θ<sub>0</sub> (i.e. the point for which we want to calculate the gradient).

___

#### Proof of the above equation:

The first step of this equation in practice is the application of the chain rule, since we consider $`f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)} (b^{(l)}))))))))`$ as the combination of $`f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (t)))))))`$ and $`a^{(l)} (b^{(l)})`$, so $`t = a^{(l)} (b^{(l)})`$:

So, since we have proved the first step, now we only have to prove that (since by the chain rule the first step of the equation is already proved) $`a^{(l)}_0 = a^{(l)} ({b_0}^{(l)})`$ and that $`∇(f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)}_0)))))))) ∗ ∇a^{(l)} ({b_0}^{(l)}) = δ^{(l)}`$:

We first prove that $`a^{(l)}_0 = a^{(l)} ({b_0}^{(l)})`$: By the definition of $`a^{(l)} (b^{(l)})`$, the latter is a function $`a^{(l)} (b^{(l)}) = W_0^{(l)} ∗ z^{(l−1)}_0 + b^{(l)}`$, but its image for b<sub>0</sub><sup>(l)</sup> transforms the variable b<sup>(l)</sup> into the constant b<sub>0</sub><sup>(l)</sup>, so $`a^{(l)} ({b_0}^{(l)}) = {W_0}^{(l)} ∗ z^{(l−1)}_0 + {b_0}^{(l)}`$, this image however coincides with the definition of a<sup>(l)</sup><sub>0</sub> (saved in the initial feedforward), since this constant is defined as the preactivation value of level l (a<sup>(l)</sup><sub>0</sub> = (a<sup>(l)</sup><sub>01</sub>, ..., a<sup>(l)</sup><sub>0nl</sub>)) of the combination defined at the beginning of point 1 (link) with input xt<sub>b</sub> = (xt<sub>b1</sub>, xt<sub>b2</sub>, ..., xt<sub>bn0</sub>) (it already has the parameters fixed on θ<sub>0</sub>), that is, given the definition of a<sup>(l)</sup>, $`a^{(l)}_0 = W_0^{(l)} ∗ z^{(l−1)} (a^{(L−1)} (…(z^{(1)} (a^{(1)} (xt_{b1}, …, xt_{bn_0}))))) + {b_0}^{(l)}`$, and since, by the definition of z<sup>(l-1)</sup><sub>0</sub> (also one of the values ​​saved in the initial feedforward), $`z^{(l-1)}_0 = z^{(l−1)} (a^{(L−1)} (…(z^{(1)} (a^{(1)} (xt_{b1}, …, xt_{bn_0})))))`$, then $`a^{(l)}_0 = {W_0}^{(l)} ∗ z^{(l−1)} (a^{(L−1)} (…(z^{(1)} (a^{(1)} (xt_{b1}, …, xt_{bn_0}))))) + {b_0}^{(l)} = {W_0}^{(l)} ∗ z^{(l−1)}_0 + {b_0}^{(l)}`$, so $`a^{(l)}_0 = a^{(l)} ({b_0}^{(l)})`$.

Now let us prove that $`∇(f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)}_0)))))))) ∗ ∇a^{(l)} (b_0^{(l)}) = δ^{(l)}`$: First, by the definition of a<sup>(l)</sup>(b<sup>(l)</sup>):

<img align=left src="https://github.com/user-attachments/assets/eb265621-7d92-43f8-80ed-27370e9812ff" width=1200>
<BR CLEAR="all"><br/>

In this case the $`z^{(l−1)}_{0i}`$ and the $`{W_0}^{(l)}_{i,j}`$ are constants, while the $`b^{(l)}_j`$ are the independent variables.

<img align=left src="https://github.com/user-attachments/assets/414f20eb-5d7f-4530-b1f1-12aa1cd67f9c" width=850>
<BR CLEAR="all"><br/>

The matrix produced is a square matrix of height nl and width nl.

At this point given that by the definition of δ<sup>(l)</sup>: $`∇(f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)}_0))))))))`$ = $`δ^{(l)}`$ = $`(δ^{(l)}_1, δ^{(l)}_2,…, δ^{(l)}_{nl})`$ = $`[δ^{(l)}_1 \, δ^{(l)}_2 … δ^{(l)}_{nl}]`$, furthermore by [this property](https://solitaryroad.com/c108.html):

<img align=left src="https://github.com/user-attachments/assets/0f0111a6-841b-451f-977f-ced43e1c61ad" width=800>
<BR CLEAR="all"><br/>

So we have proved that $`∇(f_{Cb})(z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)} (b^{(l)}))))))))`$ = $`(δ^{(l)}_1, δ^{(l)}_2,…, δ^{(l)}_{nl})`$ = $`δ^{(l)}`$, so we have proved the above equation.
___

Now, using the above equation, we derive a formula for the partial derivatives of f<sub>Cb</sub> with respect to each bias of a level l, i.e. the final formula of this point, to do so we first consider the definition of the gradient as a tuple of partial derivatives and the above equation, we also denote (to simplify the notation) the function $`f_{Cb} (z^{(L)} (a^{(L)} (z^{(L−1)} (a^{(L−1)} (…(z^{(l)} (a^{(l)} ({b_0}^{(l)}))))))))`$ with the symbol $`h(b_0^{(l)})`$:

<img align=left src="https://github.com/user-attachments/assets/de79b698-046d-4377-a3a2-66ce9598d1a6" width=800>
[EQ. 3-1]
<BR CLEAR="all"><br/>

Applying the same principles seen for obtaining the [EQ.2-2], we get that:

<img align=left src="https://github.com/user-attachments/assets/a11ff511-ee44-4232-9f72-c61dd5fdd9dd" width=500>
[EQ. 3-2]
<BR CLEAR="all"><br/>

Given the [EQ. 3-1] and [EQ. 3-2], we can substitute the tuple $`((f_{Cb})′_{b^{(l)}_{1}} (θ_0), …, (f_{Cb})′_{b^{(l)}_{nl}} (θ_0))`$ for the third term of this equation, obtaining (considering only some of the terms of [EQ. 3-1]):

<img align=left src="https://github.com/user-attachments/assets/9844638a-2995-4c80-bcb5-7be801123fe1" width=480>
<BR CLEAR="all"><br/>

So, by the definition of equality between tuples, since the first and third terms are equal, their respective components are also equal, consequently we can say that:

<img align=left src="https://github.com/user-attachments/assets/c84d646e-952f-48ab-8004-6a9cdb6a0101" width=180>
<BR CLEAR="all"><br/>

Now let's see how the backpropagation algorithm is implemented, which will use the formulas we have found up to this point to find the gradient of f<sub>C</sub> at the point θ<sub>0</sub>: <br/><br/>


### Implementation of the backpropagation algorithm:

Given the initial premises, which we rewrite below:

**Consider a feed forward network f(x, σ, θ): D x S<sub>σ</sub> x S<sub>θ</sub> -> C, a set T consisting of m pairs of the form (xt ∈ D, yt ∈ C), the set S<sub>θ</sub>, the cost function of f:**

**f<sub>C</sub>: S<sub>θ</sub> -> R, &nbsp;&nbsp; with f<sub>C</sub>(θ) = $\frac{(|f(xt_1, σ, θ) - yt_1|)^2 + (|f(xt_2, σ, θ) - yt_2|)^2 +...+ (|f(xt_m, σ, θ) - yt_m|)^2}{m}$**

**And a point θ<sub>0</sub> ∈ S<sub>θ</sub>, with:**

![image](https://github.com/user-attachments/assets/5f89d263-10ff-4d40-aa55-100898428702)

The variable θ ∈ S<sub>θ</sub> instead is:

![image](https://github.com/user-attachments/assets/6e718cef-a2a5-4ae8-8960-fc9596b92d94)

**N.B.** The point θ<sub>0</sub> and the variable θ are [k-tuples](https://en.wikipedia.org/wiki/Tuple#:~:text=In%20mathematics%2C%20a%20tuple%20is,is%20a%20non%2Dnegative%20integer.) (k is the number of total weights and biases of the neural network f).

**N.B.** The point θ<sub>0</sub> is the point that will be used for the gradient descent, so we start from a random point θ<sub>0</sub>, we calculate the gradient of f<sub>C</sub> in θ<sub>0</sub>, we continue in the direction of the gradient modifying θ<sub>0</sub> accordingly, we recalculate the gradient in the new point θ<sub>0</sub>, we continue in this way until we arrive at a point θ<sub>0</sub> that is a local minimum (or global, but it is almost impossible) such that f<sub>C</sub>(θ<sub>0</sub>) is low enough.

I note that S<sub>θ</sub> can also be written as: S<sub>θ1</sub> x S<sub>θ2</sub> x ... x S<sub>θk</sub> where S<sub>θd</sub> (with 1 <= d <= k) is the membership set of the d-th component of θ.

Now to simplify the calculations let's consider a series of cost functions, one for each element of T, more precisely let's consider m cost functions of the form:

**f<sub>Cb</sub>: S<sub>θ</sub> -> R,  &nbsp;&nbsp; with f<sub>Cb</sub>(θ) = $\frac{(|f(xt_b, σ, θ) - yt_b|)^2}{1}$ = $(|f(xt_b, σ, θ) - yt_b|)^2$**, &nbsp;&nbsp; 1 <= b <= m

**N.B.** b<sup>(l)</sup> is the bias vector of level l, while b is just a natural number between 1 and m, the two symbols have nothing to do with each other.

For the [definition of gradient](https://en.wikipedia.org/wiki/Gradient) and the properties of gradient:

<img align=left src="https://github.com/user-attachments/assets/ac8b222d-3075-446a-9d6d-9e66bd7e23cf" width=600/> [EQ. a]
<BR CLEAR="all"><br/>


Given also the final formulas found in points 1), 2) and 3):

[EQ. 1]:

<img align=left src="https://github.com/user-attachments/assets/f0d11d0f-f3ad-45f4-9f15-476ca0442358" width=1200>
<BR CLEAR="all"><br/>

[EQ. 2]:

<img align=left src="https://github.com/user-attachments/assets/076ae4ac-d45b-44a2-89a9-146c35154edd" width=250>
<BR CLEAR="all"><br/>

[EQ. 3]:

<img align=left src="https://github.com/user-attachments/assets/c84d646e-952f-48ab-8004-6a9cdb6a0101" width=180>
<BR CLEAR="all"><br/>

#### The algorithm is implemented like this:

For each training example (element xt<sub>b</sub> ∈ T):

- We perform a **feedforward**, that is, we will execute the neural network f with inputs xt<sub>b1</sub>, xt<sub>b2</sub>, ..., xt<sub>bn0</sub> and with parameters θ<sub>0</sub> and we save for each level l all the preactivation values ​​(a<sup>(l)</sup><sub>0</sub> = (a<sup>(l)</sup><sub>01</sub>, ..., a<sup>(l)</sup><sub>0nl</sub>)) and activation (z<sup>(l)</sup><sub>0</sub> = (z<sup>(l)</sup><sub>01</sub>, ..., z<sup>(l)</sup><sub>0nl</sub>)).

- We calculate **δ<sup>(L)</sup>** using [EQ. 1] and save the result, which we recycle to calculate **δ<sup>(L−1)</sup>**, which we save, and use to calculate **δ<sup>(L−2)</sup>** and so on up to **δ<sup>(1)</sup>**.

- Then, for each **W<sup>(l)</sup><sub>i,j</sub>** we use [EQ. 2] ($`δ^{(l)}_i ∗ z^{(l−1)}_{0j}`$) to compute **(f<sub>Cb</sub>)′<sub>W<sup>(l)</sup><sub>i,j</sub></sub> (θ<sub>0</sub>)** (the value $`z^{(l−1)}_{0j}`$ was saved during feedforward) and save it in a vector of length |θ<sub>0</sub>| = k called **gradient<sub>b</sub>**.

- For each **b<sup>(l)</sup><sub>i</sub>** we use [EQ. 3] ($`δ^{(l)}_i`$) to calculate **(f<sub>Cb</sub>)′<sub>b<sup>(l)</sup><sub>i</sub></sub> (θ<sub>0</sub>)** and save it in **gradient<sub>b</sub>**.

At this point, each **gradient<sub>b</sub>** contains all and only the components of the gradient **∇f<sub>Cb</sub>(θ<sub>0</sub>)** and we have a **gradient<sub>b</sub>** for each element xt<sub>b</sub> ∈ T.

Then we apply [EQ. a] to obtain the gradient **∇f<sub>C</sub>(θ<sub>0</sub>)**, and we are done.

**N.B.** b<sub>(l)</sub> is the vector of the biases of the level l, b instead is just a natural number between 1 and m, they have nothing to do with each other.
