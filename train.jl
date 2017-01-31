using Knet
using ArgParse
using JLD
using MAT
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
        ("--decayperiod"; arg_type=Int; default=0)
        ("--newoptimizer"; action=:store_true)
        ("--evalmetric"; default="bleu")
        ("--reportloss"; action=:store_true)

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

    # load data
    vocab = load(o[:vocabfile], "vocab")
    o[:vocabsize] = vocab.size
    validdata = []
    for file in o[:validdata]
        validdata = [validdata; load(file, "data")]
    end

    # initialize state and weights
    o[:atype] = !o[:nogpu] ? KnetArray{Float32} : Array{Float32}
    bestscore = o[:loadfile] == nothing ? 0 : load(o[:loadfile], "score")
    prevscore = bestscore
    w = get_weights(o)
    s = initstate(o[:atype], size(w[3], 1), o[:batchsize])
    o[:wdlen] = length(w)
    wcnn = get_wcnn(o)
    w = wcnn == nothing ? w : [wcnn; w]
    optparams = get_optparams(o, w)

    # gradient check
    if o[:gcheck] > 0
        dummy = make_batches(
            validdata[1:o[:batchsize]], vocab, o[:batchsize])[1]
        dummyimages = make_image_batches(
            validdata, dummy[1], o[:finetune])
        dummycaptions = dummy[2]
        gradcheck(loss, w, s, dummyimages, dummycaptions; gcheck=o[:gcheck])
        dummy = 0; dummyimages = 0; dummycaptions = 0; gc()
    end

    # training
    @printf("Training has been started (score=%g). [%s]\n", bestscore, now())
    flush(STDOUT)
    iter = 0
    for epoch = 1:o[:epochs]
        t0 = now()

        trainfiles = shuffle(o[:traindata])
        for k = 1:length(trainfiles)
            # make minibatching
            @printf("\nsplit: %s [%s]\n", trainfiles[k], now())
            data = load(trainfiles[k], "data")
            batches = make_batches(data, vocab, o[:batchsize])
            nbatches = length(batches)
            data = map(x->x["image"], data); gc()
            @printf("%d minibatches. [%s]\n", nbatches, now())

            # data split training
            for i = 1:nbatches
                batch = shift!(batches)
                ids, captions = batch
                images = make_image_batches(data, ids, o[:finetune])
                train!(w, s, images, captions, optparams, o)

                iter += 1
                if (o[:saveperiod] > 0 && iter % o[:saveperiod] == 0) ||
                    (o[:saveperiod] == 0 && i == nbatches)
                    @printf("\n(epoch/split/iter): %d/%d/%d [%s] ",
                            epoch, k, iter, now())
                    flush(STDOUT)
                    score, scores, bp, hlen, rlen =
                        validate(w, vocab, validdata, o)
                    @printf("\nBLEU = %.1f, %.1f/%.1f/%.1f/%.1f ",
                            100*score, map(i->i*100,scores)...)
                    @printf("(BP=%g, ratio=%g, hyp_len=%d, ref_len=%d) [%s]\n",
                            bp, hlen/rlen, hlen, rlen, now())
                    flush(STDOUT)

                    # learning rate decay
                    decay!(o, score, prevscore)
                    prevscore = score

                    # check and save best model
                    score > bestscore || continue
                    bestscore = score
                    savemodel(o, w, optparams, bestscore)
                    @printf("Model saved.\n"); flush(STDOUT)
                end
            end # batches end

            # force garbage collector
            empty!(data)
            empty!(batches)
            gc()
        end # split end

        if o[:reportloss]
            losstrn = bulkloss(w, s, o[:traindata], vocab, o)
            lossval = bulkloss(w, s, o[:validdata], vocab, o)
            @printf(
                "\nepoch:%d loss(train/val):%g/%g [%s]\n",
                epoch, losstrn, lossval, now())
        end

        t1 = now()
        elapsed = Int64(round(Float64(t1-t0)*0.001))
        @printf("\nepoch #%d finished. (time elapsed: %s)\n",
                epoch, pretty_time(elapsed))
        flush(STDOUT)
    end # epoch end
end

function decay!(o, score, prevscore)
    if !o[:adam] && score > prevscore
        @printf("\nlr decay...\n"); flush(STDOUT)
        o[:lr] *= o[:decay]
    end
end

function savemodel(o, w, optparams, bestscore)
    o[:savefile] == nothing && return

    wcopy = map(x -> convert(Array{Float32}, x), w)
    wcnn = o[:finetune] ? wcopy[1:end-o[:wdlen]] : nothing

    # optparams
    optcopy = copy_optparams(optparams, o, w)
    optcnn = o[:finetune] ? optcopy[1:end-o[:wdlen]] : nothing

    save(o[:savefile],
         "wcnn", wcnn,
         "w", wcopy[end-o[:wdlen]+1:end],
         "optcnn", optcnn,
         "optparams", optcopy[end-o[:wdlen]+1:end],
         "bestscore", bestscore)
end

function validate(w, vocab, data, o; metric=bleu)
    wcnn = o[:finetune] ? w[1:o[:wdlen]] : nothing
    wdec = o[:finetune] ? w[end-o[:wdlen]+1:end] : w
    hyp, ref = [], []
    s = initstate(o[:atype], size(w[3], 1), 1)
    for sample in data
        generation = generate(wdec, wcnn, copy(s), sample["image"], vocab)
        push!(hyp, generation)
        push!(ref, map(s->s["raw"], sample["sentences"]))
    end
    metric(hyp, ref)
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
            optparams[k] = o[:adam]?Adam(w[k];lr=o[:lr]):Sgd(;lr=o[:lr])
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
