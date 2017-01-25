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
        ("--visual"; help="visual (images|features) data file as JLD")
        ("--captions"; help="data file contains language data in JLD format")
        ("--loadfile"; default=nothing; help="pretrained model file if any")
        ("--savefile"; default=nothing; help="model save file after train")
        ("--nogpu"; action=:store_true)
        ("--hidden"; arg_type=Int; default=512)
        ("--embed"; arg_type=Int; default=512)
        ("--winit"; arg_type=Float32; default=Float32(0.01))
        ("--epochs"; arg_type=Int; default=1)
        ("--batchsize"; arg_type=Int; default=256)
        ("--lr"; arg_type=Float32; default=Float32(0.01))
        ("--gclip"; arg_type=Float32; default=Float32(5.0))
        ("--seed"; arg_type=Int; default=1; help="random seed")
        ("--gcheck"; arg_type=Int; default=0; help="gradient checking")
        ("--batchshuffle"; action=:store_true; help="shuffle batches")
        ("--finetune"; action=:store_true; help="fine tune convnet")
        ("--adam"; arg_type=Int; default=0; help="use adam optimizer")
        ("--adamlr"; arg_type=Float32; default=Float32(0.001))
        ("--decay"; arg_type=Float32; default=Float32(1.0); help="lr decay")
        ("--fast"; action=:store_true; help="do not compute train loss")
        ("--fc6drop"; arg_type=Float32; default=Float32(0.0))
        ("--fc7drop"; arg_type=Float32; default=Float32(0.0))
        ("--softdrop"; arg_type=Float32; default=Float32(0.0))
        ("--wembdrop"; arg_type=Float32; default=Float32(0.0))
        ("--vembdrop"; arg_type=Float32; default=Float32(0.0))
        ("--membdrop"; arg_type=Float32; default=Float32(0.0))
        ("--lastlayer"; default="relu7")
        ("--patience"; arg_type=Int; default=0)
        ("--saveperiod"; arg_type=Int; default=0)
        ("--improvement"; arg_type=Float32; default=Float32(0.995))
        ("--decayperiod"; arg_type=Int; default=0)
    end

    # parse args
    @printf("\nScript started. [%s]\n", now()); flush(STDOUT)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:seed] > 0 && srand(o[:seed])

    # set learning rate
    lr = o[:lr]
    decay = o[:decay]

    # set dropouts
    dropouts = Dict(
        "fc6drop" => o[:fc6drop],
        "fc7drop" => o[:fc7drop],
        "softdrop" => o[:softdrop],
        "wembdrop" => o[:wembdrop],
        "vembdrop" => o[:vembdrop],
        "membdrop" => o[:membdrop]
    )

    # load data
    visdata = load(o[:visual])
    capdata = load(o[:captions])
    vocab = capdata["vocab"]
    extradata = capdata["extradata"]

    if extradata
        visdata["train"] = vcat(visdata["train"], visdata["restval"])
        capdata["train"] = vcat(capdata["train"], capdata["restval"])
    else
        visdata["val"] = vcat(visdata["val"], visdata["restval"])
        capdata["val"] = vcat(capdata["val"], capdata["restval"])
    end
    delete!(visdata, "restval")
    delete!(capdata, "restval")

    # generate minibatches
    trn, t1, m1 = @timed make_batches(
        visdata["train"], capdata["train"], vocab, o[:batchsize])
    val, t2, m2 = @timed make_batches(
        visdata["val"], capdata["val"], vocab, o[:batchsize])
    @printf("Data loaded. Minibatch operation profiling [%s]\n", now())
    println("trn => time: ", pretty_time(t1), " mem: ", m1,
            " length: ", length(trn))
    println("val => time: ", pretty_time(t2), " mem: ", m2,
            " length: ", length(val))
    flush(STDOUT)

    atype = !o[:nogpu] ? KnetArray{Float32} : Array{Float32}
    vocabsize = vocab.size
    visual = size(trn[1][2],2)

    # initialize state and weights
    bestloss = Inf
    prevloss = Inf
    if o[:loadfile] == nothing
        w = initweights(
            atype, o[:hidden], visual, vocabsize, o[:embed], o[:winit])
    else
        w = load(o[:loadfile], "w")
        bestloss = load(o[:loadfile], "lossval")
        prevloss = bestloss
        save(o[:savefile], "w", w, "lossval", bestloss)
        w = map(i->convert(atype, i), w)
    end
    s = initstate(atype, size(w[3], 1), o[:batchsize])

    # fine tuning
    woldlen = length(w)
    if o[:finetune] && o[:cnnfile]
        wcnn = get_vgg_weights(o[:cnnfile]; last_layer=o[:lastlayer])
        w = [wcnn; w]
    elseif o[:finetune] && !o[:cnnfile]
        wcnn = load(o[:loadfile], "wcnn")
        wcnn = map(i -> convert(atype, i), wcnn)
        w = [wcnn; w]
    end


    # parameters for adam optimization
    wlen = length(w)
    optparams = Array(Any, wlen)
    for k = 1:wlen
        if o[:adam] > 0
            optparams[k] = Adam(w[k])
        else
            optparams[k] = Sgd(;lr=o[:lr])
        end
    end

    if o[:adam] > 0
        lr = o[:adamlr]
    else
        lr = o[:lr]
    end

    # gradient check
    if o[:gcheck] > 0
        gradcheck(loss, w, s, trn[1][2], trn[1][3]; gcheck=o[:gcheck])
    end

    # training
    nbatches = length(trn)
    @printf("Training has been started (lossval=%g). [%s]\n", bestloss, now())
flush(STDOUT)
saveperiod = o[:saveperiod] != 0 ? o[:saveperiod] : nbatches
best_iter = 0
patience = Inf
if o[:patience] > 0
    patience = o[:patience]
end
for epoch = 1:o[:epochs]
    t0 = now()

    for i = 1:nbatches
        iter = (epoch-1) * nbatches + i
        train!(w, s, trn[i], optparams; lr=lr, gclip=o[:gclip], dropouts=dropouts)

        if iter % saveperiod == 0
            losstrn = o[:fast] ? NaN : test(w, s, trn)
            lossval = test(w, s, val)
            @printf(
                "\nepoch/iter:%d/%d loss(train/val):%g/%g (lr: %g) [%s]\n",
                epoch, iter, losstrn, lossval, lr, now())
            flush(STDOUT)

            # learning rate decay
            if !(o[:adam] > 0 && o[:adam] <= epoch) && (lossval > prevloss)
                @printf("\nlr decay...\n"); flush(STDOUT)
                lr *= decay
            end

            # check best model
            prevloss = lossval
            if lossval > bestloss*o[:improvement]
                if div(iter-best_iter, saveperiod) > patience
                    @printf("\npatience exceed\n")
                    flush(STDOUT)
                    return
                else
                    continue
                end
            end
            bestloss = lossval
            best_iter = iter

            # save model
            if o[:savefile] == nothing
                continue
            end

            if !o[:finetune]
                save(o[:savefile],
                     "wcnn", nothing,
                     "w", karr2arr(w[end-woldlen+1:end]),
                     "lossval", bestloss)
            else
                save(o[:savefile],
                     "wcnn", karr2arr(w[1:end-woldlen]),
                     "w", karr2arr(w[end-woldlen+1:end]),
                     "lossval", bestloss)
            end
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

    if epoch == o[:adam]
        lr = o[:lr]
        @printf("switching to SGD.\n"); flush(STDOUT)
        optparams = Array(Any, wlen)
        for k = 1:wlen
            optparams[k] = Sgd(;lr=lr)
        end
    end
end
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
