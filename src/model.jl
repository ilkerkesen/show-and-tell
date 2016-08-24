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
@knet function show_and_tell(x; embed=0, vocabsize=0, fbias=0, decoding=true, o...)
    # decoding -> word emb
    # encoding -> visual emb
    if decoding
        e = wdot(x, out=embed)
    else
        e = wdot(x, out=embed)
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
        return soft(m, out=vocabsize)
    end
end
