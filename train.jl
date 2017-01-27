using Knet
using ArgParse
using JLD
using MAT
using AutoGrad

include("lib/vocab.jl")
include("lib/model.jl")
include("lib/batch.jl")
include("lib/util.jl")
include("lib/convnet.jl")

function main(args)
    s = ArgParseSettings()
    s.description = string(
        "Show and Tell: A Neural Image Caption Generator",
        " Knet implementation by Ilker Kesen [ikesen16_at_ku.edu.tr], 2016.")

    @add_arg_table s begin
        # load/save files
        ("--traindata"; nargs='+'; help="data files for training")
        ("--validdata"; nargs='+'; help="data files for validation")
        ("--vocabfile"; help="file contains vocabulary")
        ("--loadfile"; default=nothing; help="pretrained model file if any")
        ("--savefile"; default=nothing; help="model save file after train")
        ("--cnnfile"; help="CNN file")

        # model options
        ("--winit"; arg_type=Float32; default=Float32(0.01))
        ("--hidden"; arg_type=Int; default=512)
        ("--embed"; arg_type=Int; default=512)
        ("--convnet"; default="vgg19")
        ("--lastlayer"; default="relu7")

        # training options
        ("--nogpu"; action=:store_true)
        ("--epochs"; arg_type=Int; default=1)
        ("--batchsize"; arg_type=Int; default=256)
        ("--lr"; arg_type=Float32; default=Float32(0.01))
        ("--gclip"; arg_type=Float32; default=Float32(5.0))
        ("--seed"; arg_type=Int; default=1; help="random seed")
        ("--gcheck"; arg_type=Int; default=0; help="gradient checking")
        ("--batchshuffle"; action=:store_true; help="shuffle batches")
        ("--finetune"; action=:store_true; help="fine tune convnet")
        ("--adam"; action=:store_true help="use adam optimizer")
        ("--decay"; arg_type=Float32; default=Float32(1.0); help="lr decay")
        ("--fast"; action=:store_true; help="do not compute train loss")
        ("--fc6drop"; arg_type=Float32; default=Float32(0.0))
        ("--fc7drop"; arg_type=Float32; default=Float32(0.0))
        ("--softdrop"; arg_type=Float32; default=Float32(0.0))
        ("--wembdrop"; arg_type=Float32; default=Float32(0.0))
        ("--vembdrop"; arg_type=Float32; default=Float32(0.0))
        ("--membdrop"; arg_type=Float32; default=Float32(0.0))
        ("--saveperiod"; arg_type=Int; default=0)
        ("--decayperiod"; arg_type=Int; default=0)
        ("--newoptimizer"; action=:store_true)
    end

    # parse args
    @printf("\nScript started. [%s]\n", now()); flush(STDOUT)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    # random seed
    o[:seed] > 0 && srand(o[:seed])

    # TODO: load validation data only once

    o[:atype] = !o[:nogpu] ? KnetArray{Float32} : Array{Float32}
    visual = size(trn[1][2],2)

    # initialize state and weights
    bestloss = !o[:loadfile] ? Inf : load(o[:loadfile], "lossval")
    prevloss = bestloss
    w = get_weights(o, visual, vocab.size)
    s = initstate(o[:atype], size(w[3], 1), o[:batchsize])
    o[:wdlen] = length(w)
    wcnn = get_wcnn(o)
    w = (wcnn == nothing ? w : [wcnn; w])

    # optimization parameters
    optparams = get_optparams(o, w)

    # gradient check
    if o[:gcheck] > 0
        gradcheck(loss, w, s, trn[1][2], trn[1][3]; gcheck=o[:gcheck])
    end

    # training
    nbatches = length(trn)
    @printf("Training has been started (lossval=%g). [%s]\n", bestloss, now())
    flush(STDOUT)
    o[:saveperiod] = (o[:saveperiod] != 0 ? o[:saveperiod] : nbatches)
    for epoch = 1:o[:epochs]
        t0 = now()

        for i = 1:nbatches
            iter = (epoch-1) * nbatches + i
            train!(w, s, trn[i], optparams; o=o)

        if iter % saveperiod == 0
            losstrn = o[:fast] ? NaN : test(w, s, trn)
            lossval = test(w, s, val)
            @printf(
                "\nepoch/iter:%d/%d loss(train/val):%g/%g (lr: %g) [%s]\n",
                epoch, iter, losstrn, lossval, o[:lr], now())
            flush(STDOUT)

            # learning rate decay
            decay!(o, lossval, prevloss)
            prevloss = lossval

            # check best model
            lossval > bestloss && continue
            bestloss = lossval
            savemodel(w, o, bestloss, optparams)
            @printf("Model saved.\n"); flush(STDOUT)
        end
    end

    t1 = now()
    elapsed = Int64(round(Float64(t1-t0)*0.001))
    @printf("\nepoch #%d finished. (time elapsed: %s)\n",
            epoch, pretty_time(elapsed))
    flush(STDOUT)

    # shuffle batches
    o[:batchshuffle] && shuffle!(trn)
end
end

function decay!(o, lossval, prevloss)
    if !o[:adam] && lossval > prevloss
        @printf("\nlr decay...\n"); flush(STDOUT)
        o[:lr] *= o[:decay]
    end
end

function savemodel(w, o, bestloss, optparams)
    o[:savefile] == nothing && return

    wcopy = map(x -> convert(Array{Float32}, x), w)
    wcnn = o[:finetune] ? wcopy[1:end-o[:wdlen]] : nothing

    # optparams
    optcopy = copy_optparams(optparams, w, o, wdlen)
    optcnn = o[:finetune] ? optcopy[1:end-o[:wdlen]] : nothing

    save(o[:savefile],
         "wcnn", wcnn,
         "w", wcopy[end-o[:wdlen]+1:end],
         "optcnn", optcnn,
         "optparams", optcopy[end-o[:wdlen]+1:end]
         "lossval", bestloss)
end

function get_weights(o, visual, vocabsize)
    if o[:loadfile] == nothing
        w = initweights(
            o[:atype], o[:hidden], visual, vocabsize, o[:embed], o[:winit])
    else
        w = load(o[:loadfile], "w")
        save(o[:savefile], "w", w, "lossval", bestloss)
        w = map(i->convert(o[:atype], i), w)
    end
    return w
end

function get_wcnn(o)
    if o[:finetune] && o[:cnnfile]
        return get_vgg_weights(o[:cnnfile]; last_layer=o[:lastlayer])
    elseif o[:finetune] && !o[:cnnfile]
        return map(i -> convert(o[:atype], i), load(o[:loadfile], "wcnn"))
    else
        return nothing
    end
end

function get_optparams(o, w)
    if !o[:loadfile] || o[:newoptimizer]
        optparams = Array(Any, length(w))
        for k = 1:length(optparams)
            optparams[k] = o[:adam] ? Adam(w[k]; lr=o[:lr]) : Sgd(;lr=o[:lr])
        end
    elseif o[:loadfile]
        optcnn = load(o[:loadfile], "optcnn")
        if o[:finetune] && optcnn == nothing
            for k = 1:length(w)-o[:wdlen]
                optcnn[k] = o[:adam] ? Adam(w[k]; lr=o[:lr]) : Sgd(;lr=o[:lr])
            end
        end
        optparams = load(o[:loadfile], "optparams")
        optparams = o[:finetune] ? [optcnn; optparams] : optparams
    end
    return optparams
end

function copy_optparams(optparams, w, o)
    length(optparams) == length(w) || error("length mismatch (w/opt)")
    optcopy = Array(Any, length(optparams))
    for k = 1:length(optcopy)
        if typeof(optparams[k]) == Knet.Adam
            # init parameter
            optcopy[k] = Adam(zeros(size(w[k])))

            # scalar elements
            optcopy[k].lr = optparams[k].lr
            optcopy[k].beta1 = optparams[k].beta1
            optcopy[k].beta2 = optparams[k].beta2
            optcopy[k].t = optparams[k].t
            optcopy[k].eps = optparams[k].eps

            # array elements
            optcopy[k].fstm = Array(optparams[k].fstm)
            optcopy[k].scndm = Array(optparams[k].scndm)
        elseif typeof(optparams[k]) == Knet.Sgd
            optcopy[k] = Sgd(;lr=optparams[k].lr)
        end
    end
    return optcopy
end


!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
