using Knet

# Show and Tell: A Neural Image Caption Generator
# Model: Encoder -> CNN (VGG), Decoder -> RNN (LSTM)
#
# using same notation in the paper
# i: input, f: forget, o: output, n: new memory
# m: memory, c: cell, w: word embeddings
#
# for encoding x is visual features extracted from CNN
# for decoding x is one hot word fectors, later converted to embeedings
@knet function show_and_tell(x; fbias=0, vocab=0, o...)
    if !decoding
        # LSTM, visual features as input
        i = wbf2(x,m; o..., f=:sigm)
        f = wbf2(x,m; o..., f=:sigm, binit=Constant(fbias))
        o = wbf2(x,m; o..., f=:sigm)
        n = wbf2(x,m; o..., f=:tanh)
    else
        # word embeddings (w <- x)
        w = wbf(x, out=d)

        # LSTM, word embeddings as input
        i = wbf2(w,m; o..., f=:sigm)
        f = wbf2(w,m; o..., f=:sigm, binit=Constant(fbias))
        o = wbf2(w,m; o..., f=:sigm)
        n = wbf2(w,m; o..., f=:tanh)
    end

    # note that, tanh operation performed above
    c = c .* f + i .* n
    m  = c .* o

    if decoding
        return soft(m)
    end
end
