# Plain-language summary (audio/video presentation input)

We asked a simple question: if you watch how busy a piece of network
infrastructure is over time — here, individual cells of Milan's cellular
network, recorded every ten minutes — can you tell *which* cell you are
looking at just from the rhythm of its activity, the way you might recognise a
person by their handwriting? To test this, we turned each cell's raw activity
into a string of symbols (a technique called SAX) and asked a deliberately
simple pattern-learner to name the cell from a short snippet of that symbol
string, training only on earlier data and testing on later, unseen data. It
works surprisingly well: with fifty cells it picks the right one about half the
time, when random guessing would be right only one time in fifty, and it needs
only a very short snippet to do it. Importantly, the signal really lives in the
*order* of the symbols — the grammar — not just in how much traffic there is,
because scrambling the timing while keeping the traffic levels destroys most of
the accuracy. But we also found an honest catch: the plain traffic *level* by
itself is another, sometimes stronger, giveaway, so the "grammatical
fingerprint" is one of two ways to identify a cell, not the only one. A fancier
neural network did not beat the simple method. We also found that the amount of
data needed is far smaller than an earlier proposed formula predicted, so that
formula does not hold here. Finally, when the network's behaviour shifted over
Christmas, the change showed up in the grammar about two days before it showed
up in ordinary averages — an early-warning signal. The upside is passive,
cheap identification and early drift detection from data operators already
have; the downside is a privacy risk, because cells can be fingerprinted from
coarse, even encrypted-looking, activity envelopes. Every number, figure, and
line of this study was produced autonomously by an AI research agent; a human
only made accept/reject decisions and never wrote any of the technical content.
