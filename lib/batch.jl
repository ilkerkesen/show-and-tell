function make_batch(o, samples, vocab)
    images = make_images_batch(o, map(s->s[1], samples))
    captions, masks = make_captions_batch(o, map(s->s[2], samples), vocab)
    return images, captions, masks
end

function make_images_batch(o, filenames)
    images = h5open(o[:images], "r") do f
        reduce(
            (x...)->cat(o[:finetune]?4:1, x...),
            map(x->read(f,x), filenames))
    end

    batch = nothing
    if o[:finetune]
        batch = zeros(typeof(images[1]), size(images,1:3...)..., o[:batchsize])
        batch[:,:,:,1:length(filenames)] = images
    else
        batch = zeros(typeof(images[1]), o[:batchsize], size(images,2))
        batch[1:length(filenames),:] = images
    end

    return batch
end

function make_captions_batch(o, tokens, vocab)
    # captions batch
    vectors = map(t->sen2vec(vocab, t), tokens)
    longest = mapreduce(length, max, vectors)
    pad = word2index(vocab, PAD)
    captions = map(i -> pad * ones(Int, o[:batchsize]), [1:longest...])
    masks = map(i -> falses(o[:batchsize]), [1:(longest-1)...])
    for i = 1:length(tokens)
        map!(t->captions[t][i] = vectors[i][t], [1:length(vectors[i])...])
        map!(t->masks[t][i] = 1, [1:(length(vectors[i])-1)...])
    end
    return captions, masks
end
