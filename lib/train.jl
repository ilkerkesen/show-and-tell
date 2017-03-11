# one minibatch training
function train!(w, s, image, captions, opts, o)
    values = []
    gloss = lossgradient(w, copy(s), image, captions; o=o, values=values)
    update!(w, gloss, opts)
    return values
end

# for testing
function bulkloss(w, s, o, data, vocab)
    nsamples = length(data)
    ids = [1:nsamples...]
    nbatches = div(nsamples, o[:batchsize])

    total = 0.0
    nwords = 0
    newo = Dict(
        :finetune => get(o, :finetune, false)
    )
    for i = 1:nbatches
        lower = (i-1)*o[:batchsize]+1
        upper = min(lower+o[:batchsize]-1, nsamples)
        samples = data[ids[lower:upper]]
        images, captions = make_batch(o, samples, vocab)
        values = []
        loss(w, copy(s), images, captions; o=newo, values=values)
        total  += values[1]
        nwords += values[2]
        images, captions = 0, 0; gc(); flush(STDOUT)
    end
    return total/nwords
end

# oparams{T<:Number}(::KnetArray{T}; p...)=Adam()
# oparams{T<:Number}(::Array{T}; p...)=Adam()
# oparams(a::Associative; p...)=Dict(k=>oparams(v) for (k,v) in a)
# oparams(a; p...)=map(x->oparams(x;p...), a)

# Training started (nsamples=30000, nbatches=120, loss=Inf, score=0). [2017-03-12T00:57:09.813]

# (epoch/iter): 1/120, loss: 4.64231/3.83807 [2017-03-12T00:58:30.048]
# BLEU = 60.5/33.7/18.9/11.5 (BP=0.941512, ratio=0.943157, hyp_len=8263, ref_len=8761) [2017-03-12T00:58:44.928]
# Model saved to /mnt/kufs/scratch/ikesen16/data/image-captioning/trained/nic-flickr8k-vgg16-v6-nofinetune-a-iter-120.jld
