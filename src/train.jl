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
        ("--winit"; arg_type=Float32; default=Float32(0.01))
        ("--epochs"; arg_type=Int; default=1)
        ("--batchsize"; arg_type=Int; default=128)
        ("--lr"; arg_type=Float32; default=Float32(2.0))
        ("--gclip"; arg_type=Float32; default=Float32(5.0))
        ("--seed"; arg_type=Int; default=1)
        ("--gradcheck"; action=:store_true)
        ("--batchshuffle"; action=:store_true)
    end

    @printf("\nScript started. [%s]\n", now()); flush(STDOUT)

    # parse args
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:seed] > 0 && srand(o[:seed])

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

    atype = !o[:nogpu] ? KnetArray{Float32} : Array{Float32}
    visual = size(trn[1][2], 2);
    vocabsize = voc.size
    
    # initialize state & weights
    if o[:loadfile] == nothing
        w = initweights(atype, o[:hidden], visual, vocabsize, o[:embed], o[:winit])
        s = initstate(atype, o[:hidden], o[:batchsize])
    else
        w = map(i->convert(atype, i), load(o[:loadfile], "weights"));
        s = initstate(atype, size(w[3], 1), o[:batchsize])
    end

    # training
    bestloss = Inf
    @printf("Training has been started. [%s]\n", now()); flush(STDOUT)

    for epoch = 1:o[:epochs]
        _, epochtime = @timed train!(w, s, trn; lr=o[:lr], gclip=o[:gclip])
        losstrn = test(w, s, trn)
        lossval = test(w, s, val)
        @printf("epoch:%d softloss:%g/%g (time elapsed: %s) [%s]\n",
                epoch, losstrn, lossval, pretty_time(epochtime), now())
        flush(STDOUT)

        # shuffle batches
        o[:batchshuffle] && shuffle!(trn)

        # save model
        o[:savefile] != nothing && lossval < bestloss || continue
        bestloss = lossval; save(o[:savefile], "weights", karr2arr(w))
    end
end

# one epoch training
function train!(w, s, data; lr=1.0, gclip=0.0)
    for batch in data
        batch_train!(w, copy(s), batch; lr=lr, gclip=gclip)
    end
end

# one minibatch training
function batch_train!(w, s, batch; lr=1.0, gclip=0.0)
    _, vis, seq = batch
    gloss = lossgradient(w, s, vis, seq);
    gscale = lr
    if gclip > 0
        gnorm = sqrt(mapreduce(sumabs2, +, 0, gloss))
        if gnorm > gclip
            gscale *= gclip / gnorm
        end
    end

    for k in 1:length(w)
        axpy!(-gscale, gloss[k], w[k])
    end

    isa(s,Vector{Any}) || error("State should not be Boxed.")
    for i = 1:length(s)
        s[i] = AutoGrad.getval(s[i])
    end
end

# one epoch testing
test(w, s, data) = mean(map(batch -> batch_test(w, s, batch), data))

# one minibatch testing
batch_test(w, s, batch) = loss(w, s, batch[2], batch[3])

# Knet array to Float32 array conversion
karr2arr(ws) = map(w -> convert(Array{Float32}, w), ws);

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
