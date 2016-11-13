# initialize hidden and cell arrays
function initstate(atype, hidden, batchsize)
    state = Array(Any, 2);
    state[1] = zeros(batchsize, hidden);
    state[2] = zeros(batchsize, hidden);
    return map(s->convert(atype,s), state);
end


# initialize all weights of the whole network
# w[1] & w[2] => weight and bias params for LSTM network
# w[3] & w[4] => weight and bias params for softmax layer
# w[5] & w[6] => weights for visual and textual embeddings
# s[1] & s[2] => hidden state and cell state of LSTM net
function initweights(atype, hidden, visual, vocab, embed, winit)
    w = Array(Any, 6);
    input = embed;
    w[1] = winit*randn(input+hidden, 4*hidden);
    w[2] = zeros(1, 4*hidden);
    w[3] = winit*randn(hidden, vocab);
    w[4] = zeros(1, vocab);
    w[5] = winit*randn(visual, embed);
    w[6] = winit*randn(vocab, embed);
    return map(atype, w)
end


# LSTM model - input * weight, concatenated weights
function lstm(weight, bias, hidden, cell, input; encoding=false)
    gates   = hcat(input,hidden) * weight .+ bias;
    hsize   = size(hidden,2);
    forget  = sigm(gates[:,1:hsize]);
    ingate  = sigm(gates[:,1+hsize:2hsize]);
    outgate = sigm(gates[:,1+2hsize:3hsize]);
    change  = tanh(gates[:,1+3hsize:end]);
    cell    = cell .* forget + ingate .* change;
    hidden  = outgate .* tanh(cell);
    return (hidden,cell)
end

# loss function for whole network
function loss(w, s, vis, seq)
    total = 0.0;
    count = 0;
    atype = typeof(AutoGrad.getval(w[1]));

    # visual features
    x = convert(atype, vis)
    x =  x * w[5];
    (s[1], s[2]) = lstm(w[1], w[2], s[1], s[2], x);

    # textual features
    x = convert(atype, seq[1]);
    for i = 1:length(seq)-1
        x = x * w[6];
        (s[1], s[2]) = lstm(w[1], w[2], s[1], s[2], x);
        ypred = logp(s[1] * w[3] .+ w[4], 2);
        ygold = convert(atype, seq[i+1]);
        total += sum(ygold .* ypred);
        count += size(ygold, 1);
        x = ygold;
    end

    return -total / count;
end


lossgradient = grad(loss)
