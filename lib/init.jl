# initialize hidden and cell arrays
function initstate(atype, hidden, batchsize)
    state = Array(Any, 2)
    state[1] = zeros(batchsize, hidden)
    state[2] = zeros(batchsize, hidden)
    return map(s->convert(atype,s), state)
end

# initialize all weights of decoder network
# w[1] & w[2] => weight and bias params for LSTM network
# w[3] & w[4] => weight and bias params for softmax layer
# w[5] & w[6] => weights for visual and textual embeddings
# s[1] & s[2] => hidden state and cell state of LSTM net
function initweights(atype, hidden, visual, vocab, embed, winit)
    w = Array(Any, 6)
    input = embed
    w[1] = winit*randn(input+hidden, 4*hidden)
    w[2] = zeros(1, 4*hidden)
    w[2][1:hidden] = 1 # forget gate bias
    w[3] = winit*randn(hidden, vocab)
    w[4] = zeros(1, vocab)
    w[5] = winit*randn(visual, embed)
    w[6] = winit*randn(vocab, embed)
    return map(i->convert(atype, i), w)
end
