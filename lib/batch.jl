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
    return images
end

function make_captions_batch(o, tokens, vocab)
    # captions batch
    vectors = map(t->sen2vec(vocab, t), tokens)
    longest = mapreduce(length, max, vectors)
    captions = map(i->zeros(Cuchar, length(tokens), vocab.size), [1:longest...])
    for i = 1:length(tokens)
        map!(j -> captions[j][i,vectors[i][j]] = 1,
             [1:length(vectors[i])...])
    end
    return captions
end
