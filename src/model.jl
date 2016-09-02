using Knet

# Show and Tell: A Neural Image Caption Generator
# Model: Encoder -> CNN (VGG), Decoder -> RNN (LSTM)
#
# using same notation in the paper
# i: input, f: forget, o: output, n: new memory
# m: memory, c: cell, e: embeddings
# v: visual input, s: sentence input
@knet function imgcap(x; embed=0, vocabsize=0, decoding=true, fbias=0, pdrop=0.5, o...)
    # different embeddings for different input types
    if decoding
        e = wdot(x; out=embed)
    else
        e = wdot(x; out=embed)
    end

    # LSTM, embeddings as input
    i = wbf2(e,m; o..., f=:sigm)
    f = wbf2(e,m; o..., f=:sigm, binit=Constant(fbias))
    o = wbf2(e,m; o..., f=:sigm)
    n = wbf2(e,m; o..., f=:tanh)

    # note that, tanh operation performed above
    c = c .* f + i .* n
    m  = c .* o

    # word prediction
    if decoding
        d = drop(m; pdrop=pdrop)
        return wbf(d; out=vocabsize, f=:soft)
    end
end
