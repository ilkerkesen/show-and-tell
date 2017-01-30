# generate minibatches
function make_batches(data, vocab, batchsize)
    samples = []
    for k = 1:length(data)
        for s in data[k]["sentences"]
            push!(samples, (k, s["tokens"]))
        end
    end
    nsamples = length(samples)
    nbatches = div(nsamples, batchsize)
    batches = Any[]
    shuffle!(samples)

    # build batches
    for n = 1:nbatches
        lower = (n-1)*batchsize+1
        upper = min(lower+batchsize-1, nsamples)
        bsamples = samples[lower:upper]
        ids = map(x->x[1], bsamples)
        vectors = map(s -> sen2vec(vocab, s[2]), bsamples)
        longest = mapreduce(length, max, vectors)
        captions = map(
            i -> zeros(Cuchar, upper-lower+1, vocab.size), [1:longest...])

        # build captions array for neural network
        for i = 1:upper-lower+1
            map!(j -> captions[j][i,vectors[i][j]] = 1,
                 [1:length(vectors[i])...])
        end

        push!(batches, (ids, captions))
    end

    return batches
end

# make image batches
function make_image_batches(data, ids, finetune)
    mapreduce(s->s["image"], (x...)->cat(finetune ? 4 : 1, x...), data[ids])
end
