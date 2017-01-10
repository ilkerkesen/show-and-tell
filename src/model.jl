 # dropout layer
function dropout(x,d)
    if d > 0
        return x .* (rand!(similar(AutoGrad.getval(x))) .> d) * (1/(1-d))
    else
        return x
    end
end

# loss functions
function loss(w, s, visual, captions; dropouts=Dict(), finetune=false)
    if finetune
        visual = vgg19(w[1:end-6], KnetArray(visual); dropouts=dropouts)
        visual = transpose(visual)
    else
        atype = typeof(AutoGrad.getval(w[1]))
        visual = convert(atype, visual)
    end

    return decoder(w[end-5:end], s, visual, captions; dropouts=dropouts)
end

# loss gradient functions
lossgradient = grad(loss)

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
    w[3] = winit*randn(hidden, vocab)
    w[4] = zeros(1, vocab)
    w[5] = winit*randn(visual, embed)
    w[6] = winit*randn(vocab, embed)
    return map(i->convert(atype, i), w)
end

# LSTM model - input * weight, concatenated weights
function lstm(weight, bias, hidden, cell, input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

# loss function for decoder network
function decoder(w, s, vis, seq; dropouts=Dict())
    total, count = 0, 0
    atype = typeof(AutoGrad.getval(w[1]))

    # set dropouts
    vembdrop = get(dropouts, "vembdrop", 0.0)
    wembdrop = get(dropouts, "wembdrop", 0.0)
    softdrop = get(dropouts, "softdrop", 0.0)
    fc7drop  = get(dropouts, "fc7drop", 0.0)

    # visual features
    vis = dropout(vis, fc7drop)
    x = vis * w[5]
    x = dropout(x, vembdrop)

    # feed LSTM with visual embeddings
    (s[1], s[2]) = lstm(w[1], w[2], s[1], s[2], x)

    # textual features
    x = convert(atype, seq[1])
    for i = 1:length(seq)-1
        x = x * w[6]
        x = dropout(x, wembdrop)
        (s[1], s[2]) = lstm(w[1], w[2], s[1], s[2], x)
        ht = s[1]
        ht = dropout(ht, softdrop)
        ypred = logp(ht * w[3] .+ w[4], 2)
        ygold = convert(atype, seq[i+1])
        total += sum(ygold .* ypred)
        count += size(ygold, 1)
        x = ygold
    end

    return -total / count
end

# one epoch training
function train!(w, s, data, optparams; lr=0.0, gclip=0.0, dropouts=Dict(), finetune=false)
    for batch in data
        gloss = lossgradient(w, copy(s), batch[2:end]...; dropouts=dropouts, finetune=finetune)

        # gradient clipping
        gscale = lr
        if gclip > 0
            gnorm = sqrt(mapreduce(sumabs2, +, 0, gloss))
            if gnorm > gclip
                gscale *= gclip / gnorm
            end
        end

        # updateparams
        for k in 1:length(w)
            optparams[k].lr = gscale
            update!(w[k], gloss[k], optparams[k])
        end

        isa(s,Vector{Any}) || error("State should not be Boxed.")
        for i = 1:length(s)
            s[i] = AutoGrad.getval(s[i])
        end
        flush(STDOUT)
    end
end

# split testing
function test(w, s, batches; finetune=false)
    total = 0.0
    count = 0
    for batch in batches
        _, img, cap = batch
        total += loss(w, copy(s), img, cap; finetune=finetune)
        count += 1
        flush(STDOUT)
    end
    return total/count
end

# generate
function generate(w, wcnn, s, vis, vocab, maxlen; beamsize=1)
    atype = typeof(AutoGrad.getval(w[1]))
    if wcnn != nothing
        vis = KnetArray(vis)
        vis = vgg19(wcnn, vis)
        vis = transpose(vis)
    else
        vis = convert(atype, vis)
    end
    x = vis * w[5]
    (s[1], s[2]) = lstm(w[1], w[2], s[1], s[2], x)

    # language generation with (sentence, state, probability) array
    sentences = Any[(Any[SOS],s,0.0)]
    while true
        changed = false
        for i = 1:beamsize
            # get current sentence
            curr = shift!(sentences)
            sentence, st, prob = curr

            # get last word
            word = sentence[end]
            if word == EOS || length(sentence) >= maxlen
                push!(sentences, curr)
                continue
            end

            # get probabilities
            onehotvec = zeros(Cuchar, 1, vocab.size)
            onehotvec[word2index(vocab, word)] = 1
            x = convert(atype, onehotvec) * w[6]
            (st[1], st[2]) = lstm(w[1], w[2], st[1], st[2], x)
            ypred = logp(st[1] * w[3] .+ w[4], 2)
            ypred = convert(Array{Float32}, ypred)[:]

            # add most probable predictions to array
            maxinds = sortperm(ypred, rev=true)
            for j = 1:beamsize
                ind = maxinds[j]
                new_word = index2word(vocab, ind)
                new_sentence = copy(sentence)
                new_state = copy(st)
                new_probability = prob + ypred[ind]
                push!(new_sentence, new_word)
                push!(sentences, (new_sentence, new_state, new_probability))
            end
            changed = true

            # skip first loop
            if word == SOS
                break
            end
        end

        orders = sortperm(map(x -> x[3], sentences), rev=true)
        sentences = sentences[orders[1:beamsize]]

        if !changed
            break
        end
    end

    sentence = first(sentences)[1]
    if sentence[end] == EOS
        pop!(sentence)
    end
    push!(sentence, ".")
    output = join(sentence[2:end], " ")
end
