# dropout layer
dropout(x,d) = x .* (rand!(similar(AutoGrad.getval(x))) .> d) * (1/(1-d))

# loss function
function loss(ws, wadd, s, images, captions; pdrop=0.0)
    images = KnetArray(images)
    if wadd == nothing
        visual = transpose(vgg16(ws[1:end-6], images; pdrop=pdrop))
    else
        visual = transpose(vgg16(wadd, images; pdrop=pdrop))
    end
    decoder(ws[end-5:end], s, visual, captions; pdrop=pdrop)
end

# loss gradient
lossgradient = grad(loss)

# initialize hidden and cell arrays
function initstate(atype, hidden, batchsize)
    state = Array(Any, 2);
    state[1] = zeros(batchsize, hidden);
    state[2] = zeros(batchsize, hidden);
    return map(s->convert(atype,s), state);
end

# initialize all weights of decoder network
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
    return map(i->convert(atype, i), w)
end


# LSTM model - input * weight, concatenated weights
function lstm(weight, bias, hidden, cell, input)
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


# loss function for decoder network
function decoder(w, s, vis, seq; pdrop=0.0)
    total, count = 0, 0;
    atype = typeof(AutoGrad.getval(w[1]));

    # visual features
    x =  vis * w[5];
    (s[1], s[2]) = lstm(w[1], w[2], s[1], s[2], x);

    # textual features
    x = convert(atype, seq[1]);
    for i = 1:length(seq)-1
        x = x * w[6];
        (s[1], s[2]) = lstm(w[1], w[2], s[1], s[2], x)
        ht = s[1]
        if pdrop > 0.0
            ht = dropout(ht, pdrop)
        end
        ypred = logp(ht * w[3] .+ w[4], 2);
        ygold = convert(atype, seq[i+1]);
        total += sum(ygold .* ypred);
        count += size(ygold, 1);
        x = ygold;
    end

    return -total / count;
end

# generate
function generate(w1, w2, s, image, vocab, maxlen)
    atype = typeof(AutoGrad.getval(w2[1]))
    image = KnetArray(image)
    vis = vgg16(w1, image)
    x = transpose(vis) * w2[5]
    (s[1], s[2]) = lstm(w2[1], w2[2], s[1], s[2], x)

    # language generation
    word = SOS
    sentence = Any[word]
    len = 1

    while word != EOS && len < maxlen
        x = reshape(word2onehot(vocab, word), 1, vocab.size)
        x = convert(atype, x) * w2[6]
        (s[1], s[2]) = lstm(w2[1], w2[2], s[1], s[2], x);
        ypred = s[1] * w2[3] .+ w2[4]
        ypred = convert(Array{Float32}, ypred)
        word = index2word(vocab, indmax(ypred))
        push!(sentence, word)
        len += 1
    end

    if word == EOS
        pop!(sentence)
    end

    return join(sentence[2:end], " ")
end
