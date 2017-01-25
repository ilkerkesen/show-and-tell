# generate minibatches
function make_batches(visuals, captions, vocab, batchsize; finetune=false)
    catdim = finetune ? 4 : 1
    nvisuals = length(visuals)
    if nvisuals != length(captions)
        error("dimensions mismatch (visuals/captions)")
    end

    data = []
    for i = 1:nvisuals
        filename1, image = visuals[i]
        filename2, sentences = captions[i]
        filename1 == filename2 || error("filename mismatch")
        for sentence in sentences
            push!(data, (filename1, image, sentence[2]))
        end
    end

    nsamples = length(data)
    nbatches = div(nsamples, batchsize)
    batches = Any[]
    shuffle!(data)

    # build batches
    for n = 1:nbatches
        lower = (n-1)*batchsize+1
        upper = min(lower+batchsize-1, nsamples)
        samples = data[lower:upper]
        vectors = map(s -> s[3], samples)
        longest = mapreduce(length, max, vectors)

        # batch data
        bfilenames = map(s -> s[1], samples)
        bvisuals = mapreduce(s -> s[2], (x...) -> cat(catdim, x...), samples)
        bcaptions = map(
            i -> zeros(Cuchar, upper-lower+1, vocab.size), [1:longest...])

        # build captions array for neural network
        for i = 1:upper-lower+1
            map!(j -> bcaptions[j][i,vectors[i][j]] = 1,
                 [1:length(vectors[i])...])
        end

        push!(batches, (bfilenames, bvisuals, bcaptions))
    end

    return batches
end
