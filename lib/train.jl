# one minibatch training
function train!(w, s, image, captions, optparams, o)
    gloss = lossgradient(w, copy(s), image, captions; o=o)

    # gradient clipping
    gscale = o[:lr]
    if o[:gclip] > 0
        gnorm = sqrt(mapreduce(sumabs2, +, 0, gloss))
        if gnorm > o[:gclip]
            gscale *= o[:gclip] / gnorm
        end
    end

    # update parameters
    for k in 1:length(w)
        optparams[k].lr = gscale
        update!(w[k], gloss[k], optparams[k])
    end

    isa(s,Vector{Any}) || error("State should not be Boxed.")
    for i = 1:length(s)
        s[i] = AutoGrad.getval(s[i])
    end
end

# for testing
function bulkloss(w, s, o, data, vocab)
    nsamples = length(data)
    ids = [1:nsamples...]
    nbatches = div(nsamples, o[:batchsize])

    total = 0.0
    count = 0
    for i = 1:nbatches
        lower = (i-1)*o[:batchsize]+1
        upper = min(lower+o[:batchsize]-1, nsamples)
        samples = data[ids[lower:upper]]
        images, captions = make_batch(o, samples, vocab)
        total += loss(w, copy(s), images, captions; o=o)
        count += 1
        images, captions = 0, 0; gc(); flush(STDOUT)
    end
    return total/count
end
