using Knet
using ArgParse
using JLD

include("vocab.jl")
include("model.jl")
include("batch.jl")
include("util.jl")

function main(args)
    s = ArgParseSettings()
    s.description = string(
        "Show and Tell: A Neural Image Caption Generator",
        " Knet implementation by Ilker Kesen [ikesen16_at_ku.edu.tr], 2016.")

    @add_arg_table s begin
        ("--datafile"; help="data file contains dataset splits and vocabulary")
        ("--loadfile"; default=nothing; help="pretrained model file if any")
        ("--savefile"; default=nothing; help="model save file after train")
        ("--nogpu"; action=:store_true)
        ("--hidden"; arg_type=Int; default=512)
        ("--embed"; arg_type=Int; default=512)
        ("--winit"; arg_type=Float32; default=0.01)
        ("--epochs"; arg_type=Int; default=1)
        ("--batchsize"; arg_type=Int; default=128)
        ("--lr"; arg_type=Float32; default=2.0)
        ("--gclip"; arg_type=Float32; default=5.0)
        ("--seed"; arg_type=Int; default=1)
        ("--gradcheck"; action=:store_true)
        ("--batchshuffle"; action=:store_true)
    end

    @printf("\nScript started. [%s]\n", now()); flush(STDOUT)

    # parse args
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    # load data
    data = load(o[:datafile])
    voc = data["voc"]
    shuffle!(data["trn"])

    # generate minibatches
    trn, t1, m1 = @timed make_batches(data["trn"], voc, o[:batchsize])
    val, t2, m2 = @timed make_batches(data["val"], voc, o[:batchsize])
    @printf("Data loaded. Minibatch operation profiling [%s]\n", now())
    println("trn => time: ", pretty_time(t1), " mem: ", m1, " length: ", length(trn))
    println("val => time: ", pretty_time(t2), " mem: ", m2, " length: ", length(val))
    flush(STDOUT)

    # TODO: get visual features size here
    atype = !o[:nogpu] ? KnetArray : Float32
    visual = nothing;
    vocabsize = voc.size
    
    # initialize state & weights
    s = initstate(atype, o[:hidden], o[:batchsize])
    if o[:loadfile] == nothing
        w = initweights(atype, o[:hidden], visual, vocabsize, o[:embed], o[:winit])
    else
        w = load(o[:loadfile], "weights")
    end

    # training
    bestloss = Inf
    @printf("Training has been started. [%s]\n", now()); flush(STDOUT)

    for epoch = 1:o[:epochs]
        _, epochtime = @timed train(net, trn, voc; gclip=o[:gclip])
        trnloss = test(net, trn, voc)
        valloss = test(net, val, voc)
        @printf("epoch:%d softloss:%g/%g (time elapsed: %s) [%s]\n",
                epoch, trnloss, valloss, pretty_time(epochtime), now())
        flush(STDOUT)

        # shuffle batches
        o[:batchshuffle] && shuffle!(trn)

        # save model
        valloss > bestloss || o[:savefile] == nothing || continue
        bestloss = valloss; save(o[:savefile], "weights", karr2arr(w))
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
    N = size(txt,1)-1

    for j=1:N
        ygold, x = full(txt[j+1]), full(txt[j])
        ypred = (tst?forw:sforw)(f, x; decoding=true, dropout=dropout)
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

karr2arr(ws) = map(w -> convert(Array{Float32}, w), ws);

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
