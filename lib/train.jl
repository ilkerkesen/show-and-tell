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
function bulkloss(w, s, datafiles, vocab, o)
    total = 0.0
    count = 0
    for datafile in datafiles
        data = load(datafile, "data")
        batches = make_batches(data, vocab, o[:batchsize])
        for batch in batches
            ids, captions = batch
            images = make_image_batches(data, ids, o[:finetune])
            total += loss(w, copy(s), images, captions; o=o)
            count += 1
            flush(STDOUT)
        end
    end
    return total/count
end
