function make_batch(o, samples, vocab)
    images = make_images_batch(o, map(s->s[1], samples))
    x, y, batchsizes = make_captions_batch(o, map(s->s[2], samples), vocab)
    return images, x, y, batchsizes
end

function make_images_batch(o, filenames)
    images = h5open(o[:images], "r") do f
        reduce(
            (x...)->cat(o[:finetune]?4:2, x...),
            map(x->read(f,x), filenames))
    end

    batch = nothing
    if o[:finetune]
        batch = zeros(typeof(images[1]), size(images,1:3...)..., o[:batchsize])
        batch[:,:,:,1:length(filenames)] = images
    else
        # batch = zeros(typeof(images[1]), size(images,1), o[:batchsize])
        # batch[:,1:length(filenames)] = images
        batch = images
    end

    return batch
end

function make_captions_batch(o, words, vocab)
    tokens = map(wi->sen2vec(vocab, wi), words)
    longest = mapreduce(length, max, tokens)
    batchsizes = zeros(Int, longest-1)
    x = Int32[]; y = Int32[]
    for t = 1:longest-1
        for i = 1:length(tokens)
            length(tokens[i])-1 < t && break
            # @show i, t, size(tokens[i])
            push!(x, tokens[i][t])
            push!(y, tokens[i][t+1])
            batchsizes[t] += 1
        end
    end
    return x, y, batchsizes
end
