# generate minibatches
function make_batches(data, voc, batchsize)
    nsamples = length(data)
    nbatches = div(nsamples, batchsize)
    batches = Any[]

    for n = 1:nbatches
        lower = (n-1)*batchsize+1
        upper = min(lower+batchsize-1, nsamples)
        samples = data[lower:upper]
        longest = mapreduce(s -> length(s[3]), max, samples)

        # filenames
        fs = map(s -> s[1], samples)

        # visual features concat
        visual = mapreduce(s -> s[2], hcat, samples)

        # build sentences array
        sentences = map(i -> spzeros(Float32, voc.size, upper-lower+1), [1:longest...])
        for i = 1:upper-lower+1 # slice
            sen = samples[i][3]
            len = length(sen)
            for j = 1:longest # sentence
                j > len && break
                sentences[j][sen[j], i] = 1.0
            end
        end

        push!(batches, (fs, visual, sentences))
    end

    return batches
end
