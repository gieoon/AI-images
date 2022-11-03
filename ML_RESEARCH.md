- Study/learn typical decoder generation.

# Autoregressive Decoders 
https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/autoregressive_wrapper.py (top_p, top_k)
 
# Absolute Positional Embedding / Fixed Positional Embedding

Positional encoding adds variables such that vectors thata re further away from each other have a higher dot product and closer ones have a smaller dot product.

# Attention

https://dmol.pub/dl/attention.html

1. Identify which parts to attend to.
2. Extract features with high attention.

First is a search problem.
List all of the available options first.
1. Add a query. (Q)
2. The available options are the Key. (K)
3. Compute the attention mask, at each step, calculate how similar is the key, to the query. Dot product for each, to see how similar it is.
4. Extract the values based on attention. Get the values of the highest attention.

Each is an original positional embedding multiplied with a Dense Linear layer.

Take the softmax of Q dot K. This is kind of like the logits. Then, multiply this by value matrix again.
Each is connected to a linear layer, which trains weights.

Adding the linear layers makes it trainable for different sequences.

# Permutation Invariant/Equivariance

GNN's are permutation equivariant using reductions like sum or mean over neighbours on layers.

Attention layers are also permutation equivariant. This makes them good at aggregating nearest neighbour information.

# Normalizing Flows:

Use a lot of math, maybe that's why they're not very popular.
Need to know: 
1. Jacobian
2. Determinant
3. Change of variable theorem.

Solves issues of exact evaluation and inference of probability distributions in VAE's and GAN's.
Solves mode collapse, and vanishing gradients.
Explicitly learns the probability density function of real data f(x).
- Uses reversible functions.
Functions are either reversible, or the analytical inverse can be calculated.

Linear function (f(x) = x + 2) is reversible whereas exponential is not (f(x) = x^2)

- Trained using negative log likelihood loss function.
log p(x) = log p(z) + log | det(dz/dx)

# Monte Carlo

Uses multiple random sampling to obtain results.



# Bijector

Bijector functions can be used both forward and backward. Normalizing flow uses functions that have probability mass normalized (sum of all p(x) = 1)

This means that the size of the latent space is equal to the size of the feature space, and cannot be reduced to lower dimensions.

- Either injective (1 to 1), or surjective (onto)
Defined by whether it can be inversed (x can only be one value for each x)

Surjective function means mapping x onto every y. For every y, there is an x where f(x) = y.

# Jacobian Determinant (Change-of-variable formula)

if f(z) = 2x, then g(x) = x/2,
and Jacobian determinant is 1/2.
Handles the volume change to keep the probability normalized. (Same as change of variable theorem, integral, or 'area' represented by x needs to be the same across functions.)

f inverse = f^-1

f original = x = f(z)
f inverse = z = f^-1(x)
Contents of inverse is opposite of f(z)

Jacobian of f(x) = Inverse Jacobian of f^-1(x)

Aggregates the partial derivatives that are necessary for backpropagation. Use the determinant to change between variables.
Jacobian matrix collects all first-order partial derivatives of a multivariate function that can be used for backpropagation.
Jacobian determinant to change between variables, where it acts as a scaling factor between one coordinate and another (via determinant, which tells how much change of one variable is change in another.)

# Determinant

Determinant of a square matrix (equal length and height) is a scalar that provides information about the matrix.
Property of this is: 
det(A) = 1 / det(A^-1)

Determinant is a concept of space taken up by the matrix. Turn each row into a vector and the space between it is the determinant.

# Change of Variable theorem:

Integral is substitued into another integral (integral/volume of area is constant) is substituted in order to make it easier to compute and find an antiderivative.

We have distribution:
q(z)
And we have:
x = f(z)
Which creates a distribution:
p(x)

We want to figure out the relationship between q(z) and p(x).
For a single point on these probability distributions, what are their relations?

If p(z) requres more width, the height/integral needs to drop to adjust for the total area.
So, change value of x to make the area under the curve stay the same.
Take a small delta and take the distance of other function, making the integral constant.
Calculate the determinant of delta z's and delta x's.

Transposing a matrix does not change its determinant.

To change side of matrix, add 1/ to the top of it, which is equivalent to inverse.


# Prior

Some belief about a quantity before any evidence is taken into account.
Some hypothesis that the encoder makes to say 'what distribution could this latent variable z look like?'
For VAE, prior is the regularization term.

A common choice of prior is a normal gaussian distribution.

# Jensen's Inequality

Use this to take the lower bound instead fo the exact value.

# VAE

Implicitly learn data distribution.

https://jaan.io/what-is-variational-autoencoder-vae-tutorial/

https://deepgenerativemodels.github.io/notes/vae/

x = data
z = latent

p(z | x) = (p(x | z) * p(z)) / p(x)

denominator p(x) is the evidence

p(x) = integral (area under the curve) of p(x | z) * p(z) dz

This requires exponential time to compute, and is therefore intractable.

VAE associates the posterior distribution with a family of distributions q(z | x)

P(x | z) is not tractable.
So, create symmetric P(z | x) and train both in conjunction.
Goal is to find P(x | z) to be able to find P(x)

A VAE is a set of two trained probability distributions that operate on data x and latent variables z.

p(x | z) is decoder, going from latent variable to original data.

p(z) is known, because it is defined as a certain distribution.

Introduce stochasticity by using mean and standard variation, it is effectively a sampling operation over a probability distribution defined by mean and stdv. This means you can't backpropagate through it, because it's stochastic.

# Reparametrize the sampling layer of VAE:

Reparametrizing introduces epsilon, where epsilon learns stdv. And epsilon is drawn from the normal distribution.

Diverts away from stdv and mean, so both of them are fixed, and stochasticity is focused on epsilon.

KL divergence is used a sthe regularization term to introduce continuity in the data and equal distance between things.

# Why does ELBO use log?

# Energy-based models 
Use Langevin dynamics to work with high dimension datasets. It is an iterative optimization algorithm that introduces noise to the estimator as part of learning an objective function, and produces samples from a Bayesian posterior distribution (Distribution of X conditional on a bayesian distribution C (P(X | C)))

# Score-based models.
Measure the delta difference of a distribution.
This ignores an intractable normalizing constant and can be easily parameterized and calculated.
Train based on minimizing the Fisher Divergence (Measuring distance between two different probability models) between the model and the data distributions.

Produce samples via Langevin dynamics.

# Deep Generative Models

Generally fall into categories of
- Autoregressive models (ARM's)
- Variational Autoencoders (VAE's)
- Adversarial Network (GAN's)
- Normalizing Flows
- Energy-Based Models (EBM's)
- Diffusion Models

# Flow model

Maximize log-likelihood of real data x

Have to make sure inverse f^-1(z) is tractable.
The input and output dimensions have to be the same in order to be invertible. It is impossible for f(x) to be invertible if the dimensions are different.

Because constraints of f(x) are severely limited, so it's power is limited as well. we add multiple f(x)'s, and turn it into a flow model.

q(x) => G1 => p1(x) => G2 => p2(x) => G3 => p3(x) ...

By composing multiple generators, the model becomes more powerful.

Prominent flow-based models:
NICE - 2014 
NVP - 2015
GLOW - 2018

These employ Coupling layers. Each generator is a coupling layer.

Coupling layers make the determinant of the Jacobian tractable by making the determinant go against one diagonal. Otherwise, the size of the matrix might get very large very quickly.

GLOW uses 1x1 convolution layer.
Helps to learn the relationships between different channels (R,G,B)
This 1x1 convolution also has to be invertible.

# Diffusion model

Uses a UNet, which preserves data dimensionality of image, but downsamples during training.

x0 = original image
xT = isotropic gaussian noise

Loss function needs to be tractable so that the NN can optimize is.
q(x0) is the nebulous real distribution.
Sampling from this gives an image: x0 ~ q(x0) 
x0 and q(x0) are equivalent?

## Forward diffusion

q(xT | xT-1)

Add noise accoding to variance schedule 0 < beta1 < beta2 < beta3 ... betaT < 1

q(xT | xT-1) = N(xT; sqrt(1-betaT) * xT-1, betaT * identity)

Distribution is Standard Normal (N)

A normal distribution is defined by two parameters, a mean (mew) and a variance o^2 > 0

Each new slightly noisier image at timestep T is drawn from a conditional Gaussian distribution with mewT = sqrt(1 - betaT) * xT-1 and variance^2 = betaT

forward diffusion for next noisier image is a Normal distribution with mean of previous image multiplied by sqrt(1-betaT), and a variance of betaT.

I was thinking that this would create a result, but no, it creates a probability distribution from which a conditional sample can be drawn. (Conditional because the gaussian distribution is created via variance schedule)

So we sample the noise epsilon from a normal distribution (isotropic normal distribution) with mean of 0, and standard deviation as an identity matrix.
ϵ ~ N(0,I)

https://math.stackexchange.com/questions/1413985/normal-distribution-with-standard-deviation-i

An isotropic gaussian is one where the covariance matrix is represented by the simplified matrix Σ = σ^2 * I.
The covariance matrix is a measure of the join variability of two random variables. This shows the tendency in the linear relationship between variables.

A traditional gaussian distribution uses:
N(μ,Σ) 
Σ is the covariance matrix.
But this scales quadratically as the number of dimensions grows, so the covariance matrix is often restricted as Σ = σ^2 * I. where σ^2 * I is a scalar variance multiplied by an Identity matrix. This results in a covariance matrix where all dimensions are independent and where the variance of each dimension is the same. SO the Gaussian will be circular/spherical, and it is independent because the variance is consistent between dimensions.
https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic

Sample epsilon from N(0, I) and set the noisy image for this timestep, xT to sqrt(1-betaT) * xT-1 + sqrt(betaT * epsilon)

x0 -> xT (Adding noise)

Successive single step conditionals.
if data is continuous, each transition is parametrized as a diagonal gaussian.

Noise of xT = Identity covariance, losing all information about the original sample. (N(0, 1)), but also tractable and smaller than pure Gaussian distribution.

## Backward Diffusion

Calculating the conditional distribution p(xT-1 | xT) is intractable, since it requires knowing all possible distributions. That's why a NN is used to approximate this conditional probability distribution.

Use the same mean and variance for the backward gaussian distribution process.
p(xt-1 | xt) = N(xt-1; mean with NN weights (given distribution at current timestep and current timestep), and variance of NN weights given distribution at current timestep and current timestep)
The mean and variance are also conditioned on the current noise level t

The NN then learns the mean and variance of this conditional probability distribution.

The combination of p() and q() can be seen as a variational autoencoder (VAE). The variational lower bound can therefore be calculated to minimize the negative log-likelihood with respect to the ground truth data sample x0.

The ELBO is the sum of losses at each timestep. L0 + L1 + L2 ... LT

Since the sum of Gaussians is also Gaussian, we can sample from any timestep t with relation to x0 without having to apply q() (forward diffusion) repeatedly t times in order to sample xt.

q(xT | x0) = N(xT; sqrt(alpha_barT) * x0, (1-alpha_barT) * Identity matrix)

Forward diffusion for timestep t is dependent on sampling from a normal distribution for xT, with a mean of the cumulative product of all variance schedules up to now multiplied by the initial data sample, and a variance based on 1 - cumulative product of all variance schedules up to now multiplied by the identity function.

Then, move the formula around to predict the noise rather than the mean of the distribution, and loss is simply calculated by comparing the first loss with this sampled distribution and minimizing this.

# UNet

A DDPM uses a Unet because it's an autoencoder that compresses, then decompresses the image, making sure that the NN only learns the most pertinent information. It also introduced residual connections between the encoder and the decoder, greatly improving gradient flow.

This also uses an attention module (Not surprising, everything has attention in it, but it's not tokenized, so I'm curious how this works)

## Reduced variance objective

Reverse step, and forward process posterior conditioned on x0.

## Lower Variational bound

# Markov Chain
Mathematical system that defines a transition from one state into the next according to certain probabilistic rules. No matter how the process arrived at the current state, the possible future states are fixed.

So, previous step is equal to all previous steps combined.

Vision transformers superseded ConvNets.
ConvNext https://arxiv.org/abs/2201.03545
Merges Vision Transformers and ConvNets (Hybrid approach).

Novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics.
Trained on a weighted variational bound.

# DDPM's solve two problems:

probabilistic inference
generative sampling

# Eigenvector/Eigenvalue

The character of the matrix.
A matrix is a linear transformation applied to a vector.

Change is eigenvector, amount of change is the eigenvalue.

In PCA, used to decrease dimensions, diagonalize the covariance matrix of X by eigenvalue decomposition since the covariance matrix is symmetric.
Eigenvectors are principal axes/components of the data.
Dimensionality is reduced by projecting it in fewer principal directions than original.

K-means uses clustering to find the eigenvectors of a matrix.

Vector is an Eigenvector if: 
A . v = lambda . v
A is the matrix, v is the vector we are checking. lambda is the eigenvalue scalar. If one scalar linear transformation can be applied to a vector to create a matrix, it is an eigenvector.

All vectors change direction when multiplied by A. But if you find vectors v that are the same direction as Av, that is an eigenvector.

Decomposing a matrix into its eigenvalues and eigenvectors gives insight into the character of the matrix. Certain calculations become much easier when using the eigendecomposition of the matrix.

# Dot Product

Dot Product is used to calculate vector projections, vector decompositions, and determining orthogonality.
That's why, comparing two vectors is done with dot product.

# Doubly stochastic matric

Matrix of non-negative real numbers whose columns and rows each sum to 1.

# Closed form

Finite number of expressions to calculate the result.


# What is a prior?

# Density estimation
Estimate the probability distribution for x.
Many techniques are used for density estimation, but the most popular is log likelihood.
Maximizing log likelihood means finding the fit/distribution that matches a sample of x the best.

# Bayesian Inference
Assume that observed data X was generated using a latent variable Z.
Bayesian problem is about finding the posterior distribution and values of latent variable Z.
Four distributions:
1. P(Z) the prior distribution of the latent variable.
2. P (X | Z) The likelihood of X given Z.
3. P(X) The distribution of the observed data.
4. P (Z | X) The posterior distribution. Probability of Z given X.
Bayes formula = P (Z | X) = (P(X | Z) * P (Z)) / P(X)

This is hard to calculate, and requires random sampling, and slow convergence.
So, rather than chasing this posterior function with sampling, you can use a distribution function q of the latent variable Z. This is variational inference. (Function to model/map Z)

# .detach / torch.no_grads
Removes backpropagation for either the variable or whole function and tensors.

# Compression of models:
https://towardsdatascience.com/this-is-how-to-train-better-transformer-models-d54191299978

the authors compress their models in 2 ways:

- Quantization: they store the parameters at different precision formats (down to 4-bits) to save on computation time and memory footprint; and
- Pruning: they iteratively zero out 15% of the lowest magnitude parameters and then finetune again in order to reduce the number of operations and memory footprint (as weight matrices are now sparse).

# Activations

modify the output
RelU is sparse and nonlinear because fo the 0 portion. And easier to calculate.
https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0

Introduces non-linearity into the output.

A bunch of weights from different neurons are combined to switch the next layer neuron/perceptron on/off.

# ELBO (Log likelihood variational bound)
Tool that 

Exploits the concavity property of the logarithm function.
Concave function is one where sum of weighted average is larger than the log of the same weighted average, but of the logarithm of the same numbers.

Variational bound = log p(x) (Log likelihood)
log likelihood requires summing over latent variables, but some models could have extremely complex interactions between latent variables and input variables, so this is intractable to allow for performing that sum exactly.
So, a separate approximation model is used. This is an approximation of the true posterior distribution of the model (p(latent | x))

By introducing extra variables in the approximation, you can formulate a lower bound on the log likelihood. This helps to optimize the log likelihood and maximize it.
Instead of optimizing the log likelihood, push the lower bound up, and hope that log likelihood also increases.

Alternate betwen maximizing q and maximizing p.
KL divergence, if no difference, becomes log(1), which is 0, so result is 0.

A way to calculate the latent variable Z.
https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick

https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

https://nn.labml.ai/diffusion/ddpm/index.html

- Why is an optimizer necessary?

- Why is activation function necessary?

In final layer, it is to force the output into a domain. softmax for logits/probabilities, sigmoid for values in 0...1

In inner layers, activation function is used to ensure output is beyond linear, and to be able to approximate arbitrary functions.

- Linear Layer

Calculates via linear algebra, 
NN is many fully connected/linear/dense layers interspersed with activation functions.

What is the point of the bias vector?

.weight.size()
.bias.size()

Each input feature has its own weight.
But, the whole node only has one bias.

Activation function to map to more complex inputs/outputs?

- Loss functions.

Classification Error is the number of mistakes made. But this is not useful for backpropagation.
Hence, cross entropy and mean squared error.
Cross entropy can detect/magnify differences in tiny values through use of the log() function.

Without the log function, multiplying multiple probabilities makes the result very small. But, by using log, it stays in a tractable range.
The log base you are using doesn't matter as long as you are always using the same one. Python log() computes the natural log ln() log base e.

- What is entropy

High entropy means not much information?
Low entropy means a lot of information?
Skewed dataset is low entropy, and balanced dataset is high entropy?

Because the outcome is more predictable, so it's less 'surprising', and has lower entropy.
If it is balanced, then it is high entropy, because you are always surprised, and unable to predict the result.
So, requires more information, because it could be more possibilities.

- Continuous probability distributions

Probability Density Function (PDF):
Probability of a given density outcomse.

Cumulative distribution function (CDF):
The probability of a given value being less than or equal to the given outcome.

- PCA

PCA tries to find the underlying distribution z in unsupervised learning.

- VQ-VAE:
DIscrete representation, as opposed to continuous representation of VAE's.

http://adityaramesh.com/posts/dalle2/dalle2.html


https://github.com/rosinality/denoising-diffusion-pytorch

## Cross Entropy

KL divergence calculates the relative entropy between two probability distributions.
Cross entropy calculates the total entropy between two probability distributions.

Cross entropy can be used as a loss function when optimizing classification models.

It uses log() to make the output tractable, and correct when losses are incorrect.

PyTorch's CrossEntropyLoss expects raw-score logits as input and it applies a softmax function internally to convert them to probabilities.

- Deep Belief Network


- Generative models:
Typically, learn the distribution's probability density (mass function) via maximum likelihood. Typical Likelihood models = autoregressive models, normalizing flow models, energy-based models, and VAE's.
tractable normalizing constant (Latent vector as a constant, needs to be tractable)

- How does latent variable model sampling work? What other types of generative models are there and how do their decoders work?

# Support Vector Machines:

# Diffusion model questions:

If a diffusion model can skip steps for forward diffusion, why can't it skip steps going the other way? Why do you need to predict each step one at a time?
Ah, it's because x0 is unknown. But since x0 is being predicted, why can't that just be done?

Why not: p(x{0} | xT), and treat the distribution as cumulative product of variance schedule, and just remove it?


Why does the diffusion model use timestep-based positional encoding? Why is it important that the model understands its current timesteps.


Reparameterization trick is used to place the noise epsilon into the algorithm to get the mean of the distribution. Given this mean, a (conditional) image can be reproduced from this distirbution (the variance is generated via the scheduling)

When we're generating images, we're not actually generating each value, but generating a distribution, and then sampling from that distribution later on.

# Reinforcement Learning (DeepMind)

## Policy updates

Action updates through greedy selection
or, epsilon greedy selection.
Use regret to penalize for bad actions (max reward - selected reward)

epsilon noise, with probability, select a random action.
expected regret is linear.

Can we learn the policies directly (probability of an action) instead of the values

Softmax policy search:
exponentiate the action preference Ht() and divide by sum of all exponentiated action preferences.
Exponentiation is done to get a positive number.

This is not a value, it's a learnable policy parameter.

Update the policy parameters such that expected value increases.
Expected reward, given that we're following this policy.
This is a stochastic policy.

We can't sample the expectation and then take the gradient because the sample is jsut a reward, and you can't take a gradient of just a reward.

Log-likelihood trick, Gradient of expectation of a reward given a policy. REINFORCE algorithm.

gradient of expectation of reward = Summation of all actions multiplied by the probability of selecting each of these actions, multiplied by the expectation of the reward for that action, given that we have taken that specific action.

The gradient of the logarithm of X is 1 / X

This gives us an algorithm where we have the expectation, so we can then sample from it and use stochastic gradient descent.

So, after adding a learning rate and adding the theta parameters (Ascent means adding instead of subtracting)

This softmax policy uses sampled rewards, it doesn't need any value estimates.

Add step size alpha, multiplied by some reward, multiplied by gradient of selected action (regardless of the action)

case by case basis turns Identity matrix into 1.

Exploration is not explicit, it is purely due to the policy being stochastic.

Optimism in the face of uncertainty (minimizes entropy)

UCB: Estimate the upper confidence bound. And pretty sure that actual action value will be smaller or equal to the estimate of the confidence bound + an upper confidence bound.
Then, select action values greedily depending on the upper bound.
If we're very uncertain about an action, then pick it. If we're certain, don't pick it.
The uncertainty depends on the number of times the action has been selected.

This uncertainty term amplifies the reward for an action based on how uncertain the reward of it is.

i.i.d, identical and independent variables.

UCB allows us to pick a probability by estimating an upper bound using Hoffdoeing's Theorem to get a range for the value.

Logarithmic regret.