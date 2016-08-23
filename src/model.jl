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
    if !decoding
        # visual embeddings (v <- x)
        v = wdot(x, out=embed)

        # LSTM, visual embeddings as input
        i = wbf2(v,m; o..., f=:sigm)
        f = wbf2(v,m; o..., f=:sigm, binit=Constant(fbias))
        o = wbf2(v,m; o..., f=:sigm)
        n = wbf2(v,m; o..., f=:tanh)
    else
        # word embeddings (w <- x)
        w = wdot(x, out=embed)

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
        return soft(m, out=vocabsize)
    end
end
