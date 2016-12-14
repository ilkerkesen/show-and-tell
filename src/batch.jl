# generate minibatches
function make_batches(images, captions, vocab, batchsize)
    nimages = length(images)
    if nimages != length(captions)
        error("dimensions mismatch (images/captions)")
    end

    data = []
    for i = 1:nimages
        filename1, image = images[i]
        filename2, sentences = captions[i]
        filename1 == filename2 || error("filename mismatch")
        for sentence in sentences
            push!(data, (filename1, image, sentence[2]))
        end
    end

    nsamples = length(data)
    nbatches = div(nsamples, batchsize)
    batches = Any[]
    # shuffle!(data)

    # build batches
    for n = 1:nbatches
        lower = (n-1)*batchsize+1
        upper = min(lower+batchsize-1, nsamples)
        samples = data[lower:upper]
        vectors = map(s -> s[3], samples)
        longest = mapreduce(length, max, vectors)

        # batch data
        bfilenames = map(s -> s[1], samples)
        bimages = mapreduce(s -> s[2], (x...) -> cat(4, x...), samples)
        bcaptions = map(
            i -> zeros(Cuchar, upper-lower+1, vocab.size), [1:longest...])

        # build captions array for neural network
        for i = 1:upper-lower+1
            map!(j -> bcaptions[j][i,vectors[i][j]] = 1,
                 [1:length(vectors[i])...])
        end

        push!(batches, (bfilenames, bimages, bcaptions))
    end

    return batches
end
