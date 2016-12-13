using Knet
using ArgParse
using JLD
using MAT
using AutoGrad

include("vocab.jl")
include("model.jl")
include("batch.jl")
include("util.jl")
include("convnet.jl")

function main(args)
    s = ArgParseSettings()
    s.description = string(
        "Show and Tell: A Neural Image Caption Generator",
        " Knet implementation by Ilker Kesen [ikesen16_at_ku.edu.tr], 2016.")

    @add_arg_table s begin
        ("--images"; help="data file contains image data in JLD format")
        ("--captions"; help="data file contains language data in JLD format")
        ("--loadfile"; default=nothing; help="pretrained model file if any")
        ("--savefile"; default=nothing; help="model save file after train")
        ("--cnnfile"; default=nothing; help="pretrained CNN model file")
        ("--nogpu"; action=:store_true)
        ("--hidden"; arg_type=Int; default=512)
        ("--embed"; arg_type=Int; default=512)
        ("--winit"; arg_type=Float32; default=Float32(0.01))
        ("--epochs"; arg_type=Int; default=1)
        ("--batchsize"; arg_type=Int; default=16)
        ("--lr"; arg_type=Float32; default=Float32(2.0))
        ("--gclip"; arg_type=Float32; default=Float32(5.0))
        ("--seed"; arg_type=Int; default=1)
        ("--gcheck"; arg_type=Int; default=0; help="gradient checking")
        ("--batchshuffle"; action=:store_true)
        ("--finetune"; action=:store_true; help="CNN fine tuning")
        ("--dropout"; arg_type=Float32; default=Float32(0.0); help="dropout")
        ("--lastlayer"; default="relu7"; help="last layer for feature extraction")
        ("--fast"; action=:store_true; help="do not compute train loss")
    end

    # parse args
    @printf("\nScript started. [%s]\n", now()); flush(STDOUT)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:seed] > 0 && srand(o[:seed])

    # set dropouts
    pdrop = o[:dropout]

    # load data
    imgdata = load(o[:images])
    capdata = load(o[:captions])
    vocab = capdata["vocab"]
    extradata = capdata["extradata"]

    if extradata
        imgdata["train"] = vcat(imgdata["train"], imgdata["restval"])
        capdata["train"] = vcat(capdata["train"], capdata["restval"])
    else
        imgdata["val"] = vcat(imgdata["val"], imgdata["restval"])
        capdata["val"] = vcat(capdata["val"], capdata["restval"])
    end
    delete!(imgdata, "restval")
    delete!(capdata, "restval")

    # generate minibatches
    trn, t1, m1 = @timed make_batches(
        imgdata["train"], capdata["train"], vocab, o[:batchsize])
    val, t2, m2 = @timed make_batches(
        imgdata["val"], capdata["val"], vocab, o[:batchsize])
    @printf("Data loaded. Minibatch operation profiling [%s]\n", now())
    println("trn => time: ", pretty_time(t1), " mem: ", m1, " length: ", length(trn))
    println("val => time: ", pretty_time(t2), " mem: ", m2, " length: ", length(val))
    flush(STDOUT)

    atype = !o[:nogpu] ? KnetArray{Float32} : Array{Float32}
    vocabsize = vocab.size
    
    # initialize state & weights
    # w1 -> CNN | w2 -> RNN, embeddings
    if o[:loadfile] == nothing
        vggmat = matread(o[:cnnfile])
        w1 = get_vgg_weights(vggmat; last_layer=o[:lastlayer])
        w2 = initweights(
            atype, o[:hidden], size(w1[end],1), vocabsize, o[:embed], o[:winit])
    else
        w1 = map(i->convert(atype, i), load(o[:loadfile], "w1"))
        w2 = map(i->convert(atype, i), load(o[:loadfile], "w2"))
    end
    s = initstate(atype, size(w2[3], 1), o[:batchsize])

    ws, wadd = nothing, nothing
    if o[:finetune]
        ws = [w1; w2]
    else
        wadd, ws = w1, w2
    end
    
    # training
    bestloss = Inf
    @printf("Training has been started. [%s]\n", now()); flush(STDOUT)

    for epoch = 1:o[:epochs]
        _, epochtime = @timed train!(
            ws, wadd, s, trn; pdrop=pdrop, lr=o[:lr], gclip=o[:gclip])
        losstrn = o[:fast] ? NaN : test(ws, wadd, s, trn)
        lossval = test(ws, wadd, s, val)
        if o[:gcheck] > 0
            gradcheck(loss, ws, wadd, s, trn[1][2:end]...; gcheck=o[:gcheck])
        end
        @printf("epoch:%d softloss:%g/%g (time elapsed: %s) [%s]\n",
                epoch, losstrn, lossval, pretty_time(epochtime), now())
        flush(STDOUT)

        # shuffle batches
        o[:batchshuffle] && shuffle!(trn)

        # save model
        o[:savefile] != nothing && lossval < bestloss || continue
        bestloss = lossval
        if o[:finetune]
            save(o[:savefile],
                 "w1", karr2arr(ws[1:end-6]),
                 "w2", karr2arr(ws[end-5:end]))
        else
            save(o[:savefile],
                 "w1", karr2arr(wadd),
                 "w2", karr2arr(ws))
        end
    end
end

# one epoch training
function train!(ws, wadd, s, batches; lr=0.0, gclip=0.0, pdrop=0.0)
    for batch in batches
        _, img, cap = batch
        gloss = lossgradient(ws, wadd, copy(s), img, cap; pdrop=pdrop)
        update!(gloss, ws; lr=lr, gclip=gclip)
        handlestate!(s)
    end
end

# update parameters using gradient clipping
function update!(gloss, w; lr=1.0, gclip=0.0)
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
end

# handle state matrix (don't have a strong intuition)
function handlestate!(s)
    isa(s,Vector{Any}) || error("State should not be Boxed.")
    for i = 1:length(s)
        s[i] = AutoGrad.getval(s[i])
    end
end

# split testing
test(ws, wadd, s, data) = mean(map(b -> batchtest(ws, wadd, s, b), data))

# one minibatch testing
batchtest(ws, wadd, s, b) = loss(ws, wadd, copy(s), b[2], b[3])

# Knet array to Float32 array conversion
karr2arr(ws) = map(w -> convert(Array{Float32}, w), ws);

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
