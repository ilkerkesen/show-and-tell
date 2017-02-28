using Knet
using ArgParse
using JLD
using HDF5
using MAT
using JSON
using AutoGrad

include("lib/vocab.jl")
include("lib/base.jl")
include("lib/init.jl")
include("lib/model.jl")
include("lib/batch.jl")
include("lib/convnet.jl")
include("lib/train.jl")
include("lib/eval.jl")
include("lib/util.jl")
include("lib/data.jl")

function main(args)
    s = ArgParseSettings()
    s.description = string(
        "Show and Tell: A Neural Image Caption Generator",
        " Knet implementation by Ilker Kesen [ikesen16_at_ku.edu.tr], 2016.")

    @add_arg_table s begin
        # load/save files
        ("--images"; help="images JLD file")
        ("--captions"; help="captions zip file (shared by Karpathy)")
        ("--vocabfile"; help="vocabulary JLD file")
        ("--loadfile"; default=nothing; help="pretrained model file if any")
        ("--savefile"; default=nothing; help="model save file after train")
        ("--cnnfile"; help="pre-trained CNN MAT file")
        ("--extradata"; action=:store_true;
         help="use restval split for training")

        # model options
        ("--winit"; arg_type=Float32; default=Float32(0.01))
        ("--hidden"; arg_type=Int; default=512)
        ("--embed"; arg_type=Int; default=512)
        ("--convnet"; default="vgg19")
        ("--lastlayer"; default="relu7")
        ("--visual"; arg_type=Int; default=4096)
        ("--featuremaps"; action=:store_true)
        ("--cnnmode"; arg_type=Int; default=1)

        # training options
        ("--nogpu"; action=:store_true)
        ("--epochs"; arg_type=Int; default=1)
        ("--batchsize"; arg_type=Int; default=256)
        ("--lr"; arg_type=Float32; default=Float32(0.001))
        ("--gclip"; arg_type=Float32; default=Float32(5.0))
        ("--seed"; arg_type=Int; default=1; help="random seed")
        ("--gcheck"; arg_type=Int; default=0; help="gradient checking")
        ("--finetune"; action=:store_true; help="fine tune convnet")
        ("--adam"; action=:store_true; help="use adam optimizer")
        ("--decay"; arg_type=Float32; default=Float32(1.0); help="lr decay")
        ("--fast"; action=:store_true; help="do not compute train loss")
        ("--saveperiod"; arg_type=Int; default=0)
        ("--newoptimizer"; action=:store_true)
        ("--evalmetric"; default="bleu")
        ("--beamsize"; arg_type=Int; default=1)
        ("--checkpoints"; arg_type=Int; default=1)

        # dropout values
        ("--fc6drop"; arg_type=Float32; default=Float32(0.0))
        ("--fc7drop"; arg_type=Float32; default=Float32(0.0))
        ("--softdrop"; arg_type=Float32; default=Float32(0.0))
        ("--wembdrop"; arg_type=Float32; default=Float32(0.0))
        ("--vembdrop"; arg_type=Float32; default=Float32(0.0))
        ("--membdrop"; arg_type=Float32; default=Float32(0.0))
    end

    # parse args
    @printf("\nScript started. [%s]\n", now()); flush(STDOUT)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    # random seed
    o[:seed] > 0 && srand(o[:seed])

    # load vocabulary
    vocab = load(o[:vocabfile], "vocab")
    o[:vocabsize] = vocab.size

    # initialize state and weights
    o[:atype] = !o[:nogpu] ? KnetArray{Float32} : Array{Float32}
    prevscore = bestscore = o[:loadfile] == nothing ? 0 : load(o[:loadfile], "score")
    prevloss  = bestloss  = o[:loadfile] == nothing ? Inf : load(o[:loadfile], "lossval")
    w = get_weights(o)
    s = initstate(o[:atype], size(w[end-3], 1), o[:batchsize])
    o[:wdlen] = length(w)
    wcnn = get_wcnn(o)
    w = wcnn == nothing ? w : [wcnn; w]
    optparams = get_optparams(o, w)

    # get samples used during training process
    train, restval, val = get_entries(o[:captions], ["train", "restval", "val"])
    if o[:extradata]
        train = [train; restval]
    else
        val = [val; restval]
    end
    restval = 0

    # split samples into image/sentence pairs
    train = get_pairs(train)
    valid = get_pairs(val) # keep val for validation
    gc()
    const nsamples = length(train)
    const nbatches = div(nsamples, o[:batchsize])
    const saveperiod = o[:saveperiod] > 0 ? o[:saveperiod] : nbatches

    # gradient check
    if o[:gcheck] > 0
        ids = shuffle([1:nsamples...])[1:o[:batchsize]]
        images, captions = make_batch(o, train[ids], vocab)
        gradcheck(loss, w, copy(s), images, captions; gcheck=o[:gcheck])
        images
        gc()
    end

    # checkpoints
    checkpoints = []

    # training
    ids = [1:nsamples...]
    @printf("Training has been started (nbatches=%d, score=%g). [%s]\n",
            nbatches, bestscore, now())
    flush(STDOUT)
    for epoch = 1:o[:epochs]
        t0 = now()
        shuffle!(ids)

        # data split training
        losstrn = 0
        for i = 1:nbatches
            iter = (epoch-1)*nbatches+i
            lower = (i-1)*o[:batchsize]+1
            upper = min(lower+o[:batchsize]-1, nsamples)
            samples = train[ids[lower:upper]]
            images, captions = make_batch(o, samples, vocab)
            batchloss = train!(w, s, images, captions, optparams, o)
            flush(STDOUT)
            images = 0; captions = 0; ans = 0; gc()
            losstrn += batchloss

            if iter % saveperiod == 0
                lossval = bulkloss(w,s,o,valid,vocab)
                @printf("\n(epoch/iter): %d/%d, loss: %g/%g [%s] ",
                        epoch, iter, losstrn, lossval, now())
                flush(STDOUT)
                score, scores, bp, hlen, rlen =
                    validate(w, val, vocab, o)
                @printf("\nBLEU = %.1f, %.1f/%.1f/%.1f/%.1f ",
                        100*score, map(i->i*100,scores)...)
                @printf("(BP=%g, ratio=%g, hyp_len=%d, ref_len=%d) [%s]\n",
                        bp, hlen/rlen, hlen, rlen, now())
                flush(STDOUT)


                # learning rate decay
                decay!(o, lossval, prevloss)
                prevscore = score
                prevloss  = lossval
                gc()

                # check and save best model
                score >= bestscore || lossval <= bestloss || continue

                # update score and loss values
                bestscore = score >= bestscore ? score : bestscore
                bestloss = lossval <= bestloss ? lossval : bestloss

                path, ext = splitext(abspath(o[:savefile]))
                filename  = abspath(string(path, "iter-", iter, ext))
                savemodel(o, w, optparams, filename, score, lossval)
                @printf("Model saved to %s.\n", filename); flush(STDOUT)
                push!(checkpoints, filename)
                if length(checkpoints) > o[:checkpoints]
                    oldest = shift!(checkpoints)
                    rm(oldest)
                end
            end
        end # batches end

        t1 = now()
        elapsed = Int64(round(Float64(t1-t0)*0.001))
        @printf("\nepoch #%d finished. (time elapsed: %s)\n",
                epoch, pretty_time(elapsed))
        flush(STDOUT)
        gc()
    end # epoch end
end

function decay!(o, lossval, prevloss)
    if !o[:adam] && lossval > prevloss
        @printf("\nlr decay...\n"); flush(STDOUT)
        o[:lr] *= o[:decay]
    end
end

function savemodel(o, w, optparams, filename, score, lossval)
    o[:savefile] == nothing && return

    wcopy = map(x -> convert(Array{Float32}, x), w)
    wcnn = o[:finetune] ? wcopy[1:end-o[:wdlen]] : nothing

    # optparams
    optcopy = copy_optparams(optparams, o, w)
    optcnn = o[:finetune] ? optcopy[1:end-o[:wdlen]] : nothing

    save(filename,
         "wcnn", wcnn,
         "w", wcopy[end-o[:wdlen]+1:end],
         "optcnn", optcnn,
         "optparams", optcopy[end-o[:wdlen]+1:end],
         "score", score,
         "lossval", lossval)
end

function validate(w, data, vocab, o; metric=bleu, split="val")
    wcnn = o[:finetune] ? w[1:o[:wdlen]] : nothing
    wdec = o[:finetune] ? w[end-o[:wdlen]+1:end] : w
    hyp, ref = [], []
    s = initstate(o[:atype], size(w[end-3], 1), 1)

    hyp, ref = h5open(o[:images], "r") do f
        hyp, ref = [], []
        for entry in data
            image = read(f, entry["filename"])
            caption = generate(
                wdec, wcnn, copy(s), image, vocab; beamsize=o[:beamsize])
            push!(hyp, caption)
            push!(ref, map(s->s["raw"], entry["sentences"]))
        end
        hyp, ref
    end

    return metric(hyp, ref)
end

function get_weights(o)
    if o[:loadfile] == nothing
        w = initweights(o[:atype], o[:hidden], o[:visual],
                        o[:vocabsize], o[:embed], o[:winit])
    else
        w = load(o[:loadfile], "w")
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
    if o[:loadfile] == nothing || o[:newoptimizer]
        optparams = Array(Any, length(w))
        for k = 1:length(optparams)
            optparams[k] = o[:adam]?Adam(;lr=o[:lr]):Sgd(;lr=o[:lr])
        end
    elseif o[:loadfile] != nothing
        optcnn = load(o[:loadfile], "optcnn")
        if o[:finetune] && optcnn == nothing
            for k = 1:length(w)-o[:wdlen]
                optcnn[k] = o[:adam]?Adam(w[k]; lr=o[:lr]):Sgd(;lr=o[:lr])
            end
        end
        optparams = load(o[:loadfile], "optparams")
        optparams = o[:finetune] ? [optcnn; optparams] : optparams
    end
    return optparams
end

function copy_optparams(optparams, o, w)
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
