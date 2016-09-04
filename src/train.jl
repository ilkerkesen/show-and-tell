using Knet
using ArgParse
using JLD

include("vocab.jl")
include("model.jl")

function main(args)
    s = ArgParseSettings()
    s.description = "Show and Tell: A Neural Image Caption Generator implementation by Ilker Kesen, 2016. Karpathy's data used."

    @add_arg_table s begin
        ("--datafile"; help="data file contains dataset splits and vocabulary")
        ("--loadfile"; default=nothing; help="pretrained model file if any")
        ("--savefile"; default=nothing; help="model save file after train")
        ("--hidden"; arg_type=Int; default=512)
        ("--embed"; arg_type=Int; default=512)
        ("--epochs"; arg_type=Int; default=1)
        ("--batchsize"; arg_type=Int; default=128)
        ("--lr"; arg_type=Float64; default=0.2)
        ("--dropout"; arg_type=Float64; default=0.5)
        ("--gclip"; arg_type=Float64; default=5.0)
    end

    # parse args
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    # load data and generate batches
    data = load(o[:datafile])
    voc = data["voc"]
    trn = make_batches(data["trn"], voc, o[:batchsize])
    val = make_batches(data["val"], voc, o[:batchsize])
    println("Data loaded...")
    flush(STDOUT)

    # compile knet model
    if o[:loadfile] == nothing
        net = compile(:imgcap;
                      out=o[:hidden],
                      vocabsize=voc.size,
                      embed=o[:embed],
                      pdrop=o[:dropout])
    else
        net = load(o[:loadfile])
    end

    setp(net; lr=o[:lr])
    dropout = o[:dropout] > 0.0
    bestloss = Inf

    # training loop
    @printf("Training has been started...\n"); flush(STDOUT)
    for epoch = 1:o[:epochs]
        train(net, trn, voc; gclip=o[:gclip], dropout=dropout)
        trnloss = test(net, trn, voc)
        valloss = test(net, val, voc)
        @printf("epoch:%d softloss:%g/%g\n", epoch, trnloss, valloss)
        flush(STDOUT)

        # save best model
        if valloss < bestloss
            bestloss = valloss
            if o[savefile] != nothing
                save(o[:savefile], "net", clean(net))
            end
        end
    end
end


# one epoch training
function train(f, data, voc; o...)
    for batch in data
        iter(f, batch; o...)
    end
end


# one epoch forw pass
function test(f, data, voc; o...)
    sumloss, numloss = 0.0, 0
    for batch in data
        l = iter(f, batch; tst=true, o...)
        sumloss += l
        numloss += 1
    end
    return sumloss/numloss
end


# iteration for one batch forw/back
function iter(f, batch; loss=softloss, gclip=0.0, tst=false, dropout=false, o...)
    reset!(f)
    ystack = Any[]
    sumloss = 0.0
    _, vis, txt, msk = batch

    # CNN features input
    (tst?forw:sforw)(f, vis; decoding=false)

    # batch sequence length
    N = size(txt,3)-1

    for j=1:N
        ygold, cw = txt[:,:,j+1], txt[:,:,j]
        ypred = (tst?forw:sforw)(f, cw; decoding=true, dropout=dropout)
        sumloss += loss(ypred, ygold; mask=msk[:,j])
        push!(ystack, (ygold, msk[:,j]))
    end

    if tst
        return sumloss/N
    end

    # backprop
    while !isempty(ystack)
        ygold, mask = pop!(ystack)
        sback(f, ygold, loss; mask=mask)
    end

    sback(f, nothing, loss)
    update!(f; gclip=gclip)
    reset!(f; keepstate=true)
end


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

        # build sentences & masks tensors
        sentences = zeros(Float32, voc.size, upper-lower+1, longest)
        masks = zeros(Cuchar, upper-lower+1, longest)
        for i = 1:upper-lower+1 # slice
            sen = samples[i][3]
            len = length(sen)
            for j = 1:longest # sentence
                if j <= len
                    wi = word2index(voc, sen[j])
                    sentences[wi, i, j] = 0x01
                    masks[i, j] = 0x01
                else
                    wi = pad2index(voc)
                    sentences[pad2index(voc), i, j] = 0x01
                end
            end
        end

        push!(batches, (fs, visual, sentences, masks))
    end

    return batches
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
