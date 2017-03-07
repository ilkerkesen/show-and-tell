function make_batch(o, samples, vocab)
    make_images_batch(o, map(s->s[1], samples)),
    make_captions_batch(o, map(s->s[2], samples), vocab)
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
    captions = map(i->falses(o[:batchsize], vocab.size), [1:longest...])
    for i = 1:length(tokens)
        map!(j -> captions[j][i,vectors[i][j]] = 1,
             [1:length(vectors[i])...])
    end
    return captions
end
