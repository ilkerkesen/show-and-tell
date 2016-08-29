using Knet

# Show and Tell: A Neural Image Caption Generator
# Model: Encoder -> CNN (VGG), Decoder -> RNN (LSTM)
#
# using same notation in the paper
# i: input, f: forget, o: output, n: new memory
# m: memory, c: cell, e: embeddings
# v: visual input, s: sentence input
#
# FIXME: why do i have to send both input while just one of them is being used?
@knet function show_and_tell(v, s; embed=0, vocabsize=0, decoding=true, fbias=0, o...)
    if decoding
        e = wdot(s; out=embed)
    else
        e = wdot(v; out=embed)
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
        return wbf(m; out=vocabsize, f=:soft)
    end
end
