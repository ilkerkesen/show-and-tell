function train!(w, s, images, x, y, batchsizes, opts, o)
    values = []
    gloss = lossgradient(
        w, s, images, x, y, batchsizes; o=o, values=values)
    update!(w, gloss, opts)
    return values
end

function bulkloss(w, srnn, o, data, vocab)
    nsamples = length(data)
    ids = [1:nsamples...]
    nbatches = div(nsamples, o[:batchsize])

    total, nwords = 0.0, 0
    newo = Dict(:finetune => get(o, :finetune, false),
                :atype => get(o, :atype, AutoGrad.getval(typeof(w["wsoft"]))))
    for i = 1:nbatches
        lower = (i-1)*o[:batchsize]+1
        upper = min(lower+o[:batchsize]-1, nsamples)
        samples = data[ids[lower:upper]]
        images, x, y, batchsizes = make_batch(o, samples, vocab)
        values = []
        loss(w, srnn, images, x, y, batchsizes; o=newo, values=values)
        total  += values[1]
        nwords += values[2]
        images, captions, masks = 0, 0, 0; gc(); flush(STDOUT)
    end
    return total/nwords
end

# Training started (nsamples=30000, nbatches=120, loss=Inf, score=0). [2017-03-12T00:57:09.813]

# (epoch/iter): 1/120, loss: 4.64231/3.83807 [2017-03-12T00:58:30.048]
# BLEU = 60.5/33.7/18.9/11.5 (BP=0.941512, ratio=0.943157, hyp_len=8263, ref_len=8761) [2017-03-12T00:58:44.928]
# Model saved to /mnt/kufs/scratch/ikesen16/data/image-captioning/trained/nic-flickr8k-vgg16-v6-nofinetune-a-iter-120.jld
