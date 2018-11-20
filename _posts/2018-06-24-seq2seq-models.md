---
title: "Sequence-to-Sequence Models"
excerpt: "I discuss the magic behind attention-based sequence-to-sequence models, the very same models used in tasks such as machine translation."
comments: true
mathjax: true
---

Sequence-to-sequence (seq2seq) models fascinated me when I learned about recurrent neural networks. They're particularly useful for tasks such as machine translation, i.e., translating between languages using machine learning. I was very interested in learning more about these kinds of models: how they work, how to format the input and output training data, and how adding an attention mechanism helps. Similar to my [other post on backpropagation](/backpropagation), I was a bit disappointed to see that a good number of explanations of these were either buried in mathematical rigor or very hand-wavy (albeit with source code!) So I'm going to try to find a middle ground of explaining seq2seq models. First, I'll discuss recurrent neural networks and Long Short-term Memory (LSTM) Networks. After, I'll give a quick overview of the problem of machine translation and discuss seq2seq models.

# Recurrent Neural Networks

Before discussing machine translation, I need to define the primary component used in seq2seq model: recurrent neural networks. I'll assume familiarity with regular neural networks. (If you'd like a refresher, [this post I wrote](/backpropagation) describes plain neural networks, gradient descent, and backpropagation.)

## Motivation

To motivate why we need recurrent neural networks, consider modeling sequence or time-series data such as the stock price of a company over the past year, the number of people that enter a place of business in a day, or words in a sentence. All of these are time-series data: the order of the observations is critical to understanding trends.

For now let's suppose we were monitoring the stock price of "FooBar Inc." and trying to predict the next day's stock value so we can decide whether to buy or sell. We can collect the stock price of FooBar Inc. for a number of years to use as our training and testing data. One thought might be to apply linear regression or some other regression model. This might work, depending on how complex the time-series data is, but the model is usually not linear or nice in any way.

Instead, we can try to use a neural network to predict the output values. To start, we can try to use a single stock value to predict the next one, e.g., use the average value of the stock on Monday to predict the average value of the stock on Tuesday. (Let's assume that our data is averaged for each day; in reality, stocks change more frequently than this, but, pedagogically, it's easier to explain using days!) Then we can compare our prediction with the actual value of the stock on Tuesday and train our network to minimize that difference. Then we can try to use the (actual) value of the stock on Tuesday to predict the value of the stock on Wednesday and so on. And done! Right?

![ANNs for Sequences](/images/seq2seq/ann-sequence.svg "ANNs for Sequences")

Not quite! Trying to predict using this approach won't produce good values because we're only looking one day in the past. If we maybe considered each of the stock values for the previous 5 days, we would have more _context_. This _context_ is the key! Now instead of our network taking a single input and producing a single output, we take 5 inputs, i.e., the stock of FooBar Inc. for the past 5 days, and produce a single output, i.e., the predicted stock for the next day. In this approach, shown in the figure above, we have a _context window_ of 5 days. This produces better results because we're considering more previous data before making a decision. 

The question you may have thought of is "why not have a really large context window then? If some context helps, then why don't we give all of the context? Instead of the past 5 _days_, let's give the past 5 _months_ of data." But think about what this does to our input layer. We went from an input layer of 5 to ~151 now! Our input layer just increased by 2 orders of magnitude along with the weight matrices!

Even with our handling of the input sequence using a plain neural network, the approach still isn't tuned to handling the nature of sequential data. We're treating each input datum in the context window independently to produce the output, which isn't any better than each input datum being independent in the input sequence. This doesn't model the sequence data then: an input at time $t$, i.e., $x_t$ is influenced by $x_{t-1}$, which is influenced by $x_{t-2}$ and so on. Hence, $x_t$ is really influenced by all of the previous time steps, which is not something we can model using our feedforward neural network structure. For this, we need a **recurrence relation** and along comes **recurrent neural networks** or **RNNs**!

## Vanilla Recurrent Neural Networks

![Vanilla RNNs](/images/seq2seq/vanilla-rnn.svg "Vanilla RNNs")

A **recurrent neural network** or **RNN** is a kind of neural network that specializes in processing sequence data, particularly sequences of vectors. (We call these "vanilla" or "plain" RNNs since they are the most fundamental design of RNNs; we'll see more complicated models later.) There are many different flavors of RNNs, but the most general one, shown in the figure above, takes in a sequence of inputs and produces a sequence of outputs for each input. (There are RNNs that take in a _sequence_ of inputs and produce a _single_ output for tasks like sentiment classification; there are also RNNs that do the opposite.) Mathematically, plain RNNs are quite simple: they can be described using only the following two equations.

$$
h_t = \tanh(W^{(xh)}x_t + W^{(hh)}h_{t-1} + b^{(h)})\\
\hat{y_t} = W^{(hy)}h_t + b^{(y)}
$$

where

- $x_t$ is the input vector at time $t$
- $h_t$ is the hidden state at time $t$
- $\hat{y_t}$ is the output vector at time $t$

We have a total of 5 learnable parameters: the weights from the input to the hidden layer $W^{(xh)}$, the weights from the previous hidden layer to the current hidden layer $W^{(hh)}$, the weights from the hidden layer to the output layer $W^{(hy)}$, and hidden and output biases $b^{(h)}$ and $b^{(y)}$, respectively. Notice that for each time step, we re-use these parameters. Additionally, we have one important hyperparameter: the size of the hidden layer, i.e., the length of the hidden state vector.

The key to handling sequential data, and the essence of the RNN, is the recurrence relation, i.e., the first equation, that computes the current hidden state as a function of the current input and previous hidden state. (The initial hidden state $h_0$ is usually initialized to the zero vector.) This means that the hidden state at time $t$ has all of the previous time steps $t-1, \dots,1$ folded into it: $h_t$ is a representation of everything that has been seen up to and including time step $t$.

![Backpropagation through Time (BPTT)](/images/seq2seq/bptt.svg "Backpropagation through Time (BPTT)")

So how do we train recurrent neural networks? Using backpropagation! They are _neural networks_ after all! We can represent the equations above as a computation graph and perform backpropagation to train our learnable parameters! As shown in the figure above, to get a loss value, we run the entire sequence through our network and sum the loss values at each step; then we can backpropagate with the combined loss. This is sometimes called **backpropagation through time (BPTT)**, but it's really just backpropagation.

![Truncated BPTT](/images/seq2seq/truncated-bptt.svg "Truncated BPTT")

However, if the sequence is long, backpropagation may take a long time since computational cost is directly tied to the sequence length. Instead, we window the input sequence and compute loss values and backpropagate only for values in that window. This is called **truncated backpropagation through time**, as shown in the figure above. We only compute the loss values and backpropagate for the size of the BPTT window, which is 5 in the figure above. However, when we shift over a time step, we still push forward the values of the hidden layer. We keep doing this until we reach the end of the sequence and have accumulated all of the gradients of our shared parameters over a batch; then we can perform a parameter update and train our network!

As a quick aside, we can construct deep RNNs by simply stacking the hidden layers together, similar to a feedforward neural network.

![Deep RNN](/images/seq2seq/deep-rnn.svg "Deep RNN")

In this example, the second hidden layer receives the hidden state from the first hidden layer as its input sequence. Then the third hidden layer receives the hidden state from the second hidden layer and so on until the output layer.

## Long Short-Term Memory (LSTM) Cells

Before I try to demystify the inner workings of an LSTM cell, I'll describe some issues with vanilla RNNs so we can understand why LSTM cells are favored over them.

The largest issue with vanilla RNNs is the **vanishing gradient problem**. This is the same problem that plagued feedforward neural networks: as we stack layers, the gradient computation requires multiplying more and more factors which drives the gradient down to 0, and the earlier layers don't receive much gradient, which prevents their parameters from being updated. (This assumes we're using an activation function whose gradient has an absolute value less than 1.) With RNNs, instead of _depth_ causing the gradient to vanish, the _number of time steps_ drives the gradient to zero. The more time steps we have, the more terms we multiply to compute our gradient because of the recurrence relation. By the time the earlier states receive the gradient, it's nearly gone! Intuitively, this means our network is unable to learn long-term dependencies and relationships. One preventative measure to mitigate this problem is to use rectified linear units (ReLUs) instead of sigmoid or tanh neurons since their derivative is either 0 or 1: the gradient is deleted or retained in its entirety.

A complimentary issue more common with RNNs is the **exploding gradient problem**: the gradient accumulates and drives to infinity because of the compounding effects of accumulating gradient for a number of time steps. The recurrence relation is to blame again! In practice, the gradient will quickly overflow its container (usually a 32-bit float) and cause `NaN` values to propagate through our network. A quick and simple solution is to clip the gradient at each time step. Empirically, clipping the gradient to be in the range $[-5, 5]$ works well.

A better approach to solve both of these problems is to redesign the entire RNN cell from scratch to take the gradient into consideration. This is exactly the thought with the long short-term memory (LSTM) cell.

![LSTM cell](/images/seq2seq/lstm.svg "LSTM cell")

The LSTM cell replaces the internal workings of the RNN to allow the gradient to flow more easily to the earlier states. Instead of the two equations that define the vanilla RNN, the following seven equations define the LSTM cell.

$$
f_t = \sigma(W^{(f)}[h_{t-1}; x_t] + b^{(f)})\\
i_t = \sigma(W^{(i)}[h_{t-1}; x_t] + b^{(i)})\\
g_t = \tanh(W^{(g)}[h_{t-1}; x_t] + b^{(g)})\\
o_t = \sigma(W^{(o)}[h_{t-1}; x_t] + b^{(o)})\\
C_t = f_t \odot C_{t-1} + i_t \odot g_t\\
h_t = o_t \odot \tanh(C_t)\\
\hat{y_t} = W^{(y)}h_t + b^{(y)}
$$

where

- $[h_{t-1}; x_t]$ is the concatenation of the previous hidden state and the current input
- $f_t$ is the **forget gate** values at time $t$
- $i_t$ is the **input gate** values at time $t$
- $g_t$ is the **candidate gate** values at time $t$
- $o_t$ is the **output gate** values at time $t$
- $C_t$ is the **cell state** at time $t$
- $a \odot b$ is the Hadamard product, i.e., element-wise multiplication
- $h_t$ is the hidden state at time $t$
- $\hat{y_t}$ is the output vector at time $t$
- $\sigma(\cdot)$ is the sigmoid activation function $\displaystyle\frac{1}{1+e^{-x}}$

We have more complexity in the internal cell representation that leads to interesting and desirable functionality. One thing you may notice is that we have two internal states: the hidden state and the cell state. I'll reference this a bit later so hold on to this interesting change. The LSTM introduces the concept of **gates**: we have the forget gate, input gate, candidate gate, and output gate, each with their own set of learned weights and biases.

The purpose of the **forget gate** is to forget/remove information from the previous cell state. Consider the equation where we compute the cell state $C_t$. In the first part, we take the Hadamard product of the previous cell state and values of the forget gate $f_t \odot C_{t-1}$. Since we're using a sigmoid activation function, each of the components in the forget gate will be between 0 and 1. When a component of $f_t$ is close to 0 and we take the Hadamard product, we'll zero out the corresponding component of the cell state. Intuitively, this corresponds to "forgetting". Hence, the first part of the cell state update equation tells us what to "forget" from the previous cell state.

The next two gates are related to each other. The **input gate** intuitively tell us what components of the input to keep, i.e., the components of $i_t$ close to 1, and what to remove, the components of $i_t$ close to 0. The **candidate gate** has a bit of a strange name, but it corresponds to what information to retain or keep from the input. This is why we apply the hyperbolic tangent function: because we can encode information in the range $[-1, 1]$. We take the Hadamard product of these two gates and add them to the cell state. These operations intuitively correspond to adding information to the cell state.

The final gate is the **output gate**, and it is used to update the hidden state with the cell state. It's a similar rationale to input and candidate gates: we figure out which portion of the input, which has the previous hidden state concatenated to it, to retain, and hyperbolic tangent of the new, updated cell state squashes the values to $[-1, 1]$. This incorporates the cell state into the hidden state.

Now that we have an understanding of the internal structure of the cell state, let's see if the added complexity is worth it. First, let's look at the vanishing gradient problem. Recall that it occurred because we're performing many operations on the hidden state, which led to a string of factors to multiply together when computing the gradient. However, in an LSTM, we have both a hidden state and a cell state. The cell state is only operated on through a Hadamard product and addition; the addition simply copies the gradient over, and the Hadamard product just multiplies it by a factor. These simple operations allow the gradient to travel, mostly untouched, through the cell state. Empirically, this allows LSTMs to remember long-term dependencies better than plain RNNs. As for the exploding gradient problem, clipping the gradient is still used.

As a quick aside, there are other types of cell architectures, such as the Gated Recurrent Unit (GRU), that improve upon the LSTM cell, but the LSTM cell is the basis that we'll be using for the rest of this post.

# Word Embeddings

So far, we've only discussed continuous-valued, numerical data. But for machine translation, we'll be working with words, which are neither continuos nor numerical. We need to convert words to numerical values that we can feed into our RNN. The simplest way to do this is to create a **vocabulary**: a list of all possible words in our training data. We order this vocabulary in some way (the ordering doesn't matter as long as we're consistent!) and we simply use the index of the word we're looking up. This provides a cheap way to convert between words and numbers. For example, if our vocabulary was ["the", "cat", "sat", "on", "mat"], then we can convert "the cat sat on the mat" into "0, 1, 2, 3, 0, 4". However, we usually use vectors instead of scalars for our input so we use a one-hot vector where the length/dimensionality of the vector is the same as the size of the vocabulary. Each element of the vector is 0 except we set the element of the given index to 1, hence the name **one-hot vector**. So our example sentence would be converted into six 5-dimensional vectors, one for each word. The first 5-dimensional vector would be $[1~0~0~0~0]^T$ since the index is 0. Similarly, the second would be $[0~1~0~0~0]^T$. 

If our training set is really large, then we may end up with a vocabulary of millions of words! This means our input one-hot vector would be a million-dimensional vector! This would drastically slow down our computation time. Instead, we fix our vocabulary size and replace any words that are not in our vocabulary with a special vocabulary token `<UNK>` that means unknown. (In practice, this the first vocabulary word, i.e., word with index 0.)

We also add two extra tokens: start-of-sentence (`<SOS>`) and end-of-sentence (`<EOS>`). (Again, in practice, these become the second and third vocabulary words after `<UNK>`.) These are required since the input and output sentences may have a different length. When we're loading in training data, we attach these on to the beginning and ending of the sentence as helpful markers.

One-hot encoded vectors are called **sparse vectors** because many of the vector components are zero. In fact, all but one of the vector components are zero! In practice, **dense vectors** perform better because they can represent a more complicated vector space because they have more degrees of freedom to do so. To create dense vectors from sparse vectors, we usually perform a matrix multiplication again a weight matrix, called the **embedding matrix**, and train that weight matrix using backpropagation. Since the inputs to this matrix multiplication are one-hot vectors, an equivalent operation is simply selecting a column of the embedding matrix. Now we can feed these dense vectors into the RNN!

# Machine Translation

Before jumping right into seq2seq models, I'd like to take a second to describe the problem of machine translation as well as define some terminology we'll use. **Machine translation** is the task of using computers to generate translations between languages. Think of Google Translate: we select **source** and **target languages** as well as a **source sentence**, and Google Translate produces a **target sentence** translation in the target language. Because of the complexity of languages, e.g., verb conjugations, noun declensions, and grammar, we cannot simply do a one-to-one lookup. Earlier approaches were slightly smarter than this: they stemmed words, i.e., removed any prefixes/suffixes, and used parse trees, i.e., trees that show the function of each word in the sentence, to produce the translation. Modern techniques use neural networks and seq2seq architectures.

The fundamental concept behind machine translation is **alignment**. We want to _align_ each word of the input sentence with a word in the output sentence.

![Alignment](/images/seq2seq/alignment.svg "Alignment")

Above is an example of an alignment, where white indicates a stronger alignment and black indicates a weaker alignment. This a fairly simple alignment, but, depending on the language, our alignment may not be so linear. For example, in Spanish, if I wanted to say "the red car", I would say "el coche rojo". Notice how the adjective "red" comes _after_ the noun, not before it like in English. In this case, the alignment wouldn't be so straightforward.

In another complex case, we might not get a simple one-to-one alignment. Consider "How are you?" in English being translated to "¿Como esta?" in Spanish. In this case, the correct action is the align "how" with "como" and "are you" with "esta" since it's a conjugated Spanish verb that encapsulates the "you".

Attention mechanisms used in seq2seq models inherently generate this alignment through the attention weights themselves. We'll discuss more about attention mechanisms soon.

# seq2seq Models

Now that we're familiar with RNNs and machine translation, we can try to apply one of them to the task of machine translation, in particular the one-produces-one many-to-many way of using them. This seems to work for translating something like "How are you?" in English to "Comment vas tu?" in French. However, the critical flaw is that the source and target sentences might not be the same length. For example, "How are you?" in English can be translated simply to "¿Como esta?" in Spanish, which is only two words. How should we align these if we're using a one-produces-one many-to-many model? We would have to map one of the words to some null/empty representation. But, as we've seen before, this isn't the correct alignment!

This example illustrates the point that we need to use a more powerful architecture to create this alignment: the **encoder-decoder architecture**. As the name implies, we have two major components: the encoder and the decoder. The purpose of the encoder is to take the source sentence and condense it into a single vector representation. The decoder then consumes this vector and generates the target sentence.

## Encoder

![Encoder](/images/seq2seq/encoder.svg "Encoder")

The first part of the encoder-decoder architecture is the **encoder**. This is an RNN that consumes the source sentence, including the `<SOS>` and `<EOS>` tags, and produces no outputs. We use the final hidden state because, at that point, we have _encoded_ the entire source sentence. This is sometimes called the **context vector** because it retains the context of the source sentence.

## Decoder

![Decoder](/images/seq2seq/decoder.svg "Decoder")

The other part is the **decoder**. This is also an RNN, but its job is to generate the target sentence. However, we don't initialize the initial hidden state to the zero vector as we would with any other RNN, we use the final hidden state of the encoder. This _transfers the context_ of our source sentence to our decoder.

There are two modes to the decoder: training and decoding. The training mode is used to train the entire encoder-decoder architecture, completely end-to-end. The decoding mode is used after our model has trained to generate sentences during test time.

## Training

![Training](/images/seq2seq/training.svg "Training")

We train the encoder-decoder architecture jointly: the gradient flows from the output of the decoder, where the loss is computed, to the input weights of the encoder, where the source sentence was consumed. To train, we feed the source sentence into the encoder to create the context vector; then we feed in the target sentence into the decoder, initialized with the context vector from the encoder. Then we can compute the loss from this decoder at each time step of the output, sum the losses, and backpropagate!

## Decoding at Test time

Training the encoder-decoder architecture is slightly different than evaluating it at test-time. Recall that during training, we knew the correct target sentence and fed that into the decoder. However, during testing, we no longer have the correct target sentence so we need to generate it using a **decoding algorithm**. Just like with training, we feed the source sentence into the encoder and pass the context vector to the decoder. Now we need to change how we use the decoder so we can _generate_ a target sentence.

![Greedy Sampling](/images/seq2seq/decoder.svg "Greedy Sampling")

The simplest way to generate a target sentence is to use **greedy sampling**, shown in the figure above. We feed the source sentence into the encoder and produce the context vector to pass to the decoder. Then, at the first time step, the decoder, using the context vector, will generate a softmax distribution over the vocabulary, and we can select the most likely word. We use this maximum likelihood word as the input of the next time step, using its embedding vector. Then, at the second time step, the input is the maximum likelihood output of the first time step, and we do the same thing: use the maximum likelihood word of this time step as the input to the third time step. We keep doing this until we generate the final `<EOS>` character (or hit a maximum word limit). Then we've generated the entire target sentence and can evaluate it against the ground-truth sentence! (We have to use more complicated metrics at test time though!)

![Beam Search](/images/seq2seq/beam-search.svg "Beam Search")

A better way to do this is using **beam search**. The major detriment to greedy sampling is that decisions are irreversible: if we predict the incorrect word at the beginning of the sentence, then we've made the entire sentence incorrect and cannot recover from that. To remedy this, instead of keeping just the most likely sequence, we retain the top $k$ likely sequences, where $k$ is a hyperparameter called the **beam width**.

Suppose our beam width was 5. Then, at the first time step, we keep the top 5 most likely first words. Then, at the next time step, we evaluate the top 5 most likely next words. But now we have 25 sequences so we use a language model to compute the likelihood of each of those sequences and retain only the most likely 5. We keep doing this until we've generate the `<EOS>` symbol. This way, we only have to keep 5 sequences in consideration and evaluate 25 possible ones against the language model, regardless of the length of the sequence! Using beam search, if we make a mistake early on, we still have other sequences we can consider, and the incorrect one will be trimmed!

## Attention Mechanism

![Attention Mechanism](/images/seq2seq/attention.svg "Attention Mechanism")

A state-of-the-art improvement we can make on our sequence-to-sequence model that has worked very well in the past few years is the notion of **attention**. The major issue with the encoder-decoder architecture is the **bottleneck problem**: we have to encode the entirety of the source sentence into a fixed-length vector. In other words, it must retain all of the information of the source sentence.

Instead, it would be more useful for the decoder if it could look back at the entire source sentence for reference; however, we should try to be more efficient and direct its gaze towards "relevant" parts of the source sentence. This kind of direction is called **attention**.

The details of the attention mechanism seem kind of complicated, but we can distill down this information into a sequence of a few steps.

1. For each decoding step do:
2. Take the dot product of the hidden state of the decoder and each encoder hidden state to produce attention scores
3. Take the softmax of all of the attention scores to produce an attention distribution
4. Take the weighted sum of the encoder hidden states using the attention distribution as the weights to produce the **context vector**.
5. Concatenate the **context vector** with the decoder input and produce the $\hat{y}$.

This attention mechanism improves the quality of the decoded sentence because we get access to the entire source sentence through the attention mechanism. In other words, this solves our bottleneck problem! Additionally, this also helps with our vanishing gradient problem because the attention distribution is over the entire encoder states instead of passing through the bottleneck past all of the previous decoder steps. In other words, we get a quicker path to the earlier time steps.

![Alignment](/images/seq2seq/alignment.svg "Alignment")

Additionally, we get alignments for free. We can look at the softmax distribution of the attention scores for each decoder time step.

The primary drawback of the attention mechanism is the computational complexity of the process. For each decoding step, we look through the entire input sentence. This is quite computationally expensive to do, though the results tend to be better than not using it.

# Conclusion

Recurrent neural networks are used for sequence models because the better represent the underlying sequence data they ingest. Using two of them, we can create an encoder-decoder seq2seq model that can translate between sequences, fundamentally. We've only discussed machine translation as an application of seq2seq models, but they can really be used for any sequence-to-sequence task. A recent trend in designing neural architectures is to have an attention mechanism to focus our network to a specific subsection of our input. These work very well, especially with seq2seq models, however, at a higher computation cost.

Hopefully this post helps demystify how sequence-to-sequence models and attention architectures work!
