using Knet, ArgParse
using MAT, JSON

include("vocab.jl");
include("data.jl");
include("train.jl");
include("model.jl")

function main(args)
    s = ArgParseSettings()
    s.description = "Show and Tell: A Neural Image Caption Generator implementation by Ilker Kesen, 2016. Karpathy's data used."

    @add_arg_table s begin
        ("--vggfile"; help="MAT file contains VGG features")
        ("--jsonfile"; help="JSON file contains text data")
        ("--loadfile"; default=nothing; help="pretrained model file if any")
        ("--savefile"; default=nothing; help="model save file after train")
        ("--hidden"; arg_type=Int; default=256)
        ("--embed"; arg_type=Int; default=256)
        ("--epochs"; arg_type=Int; default=10)
        ("--batchsize"; arg_type=Int; default=128)
        ("--seqlength"; arg_type=Int; default=100)
        ("--decay"; arg_type=Float64; default=0.9)
        ("--lr"; arg_type=Float64; default=0.2)
        ("--gclip"; arg_type=Float64; default=5.0)
    end

    # parse args
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o)

    # build vocabulary and split data
    data, voc, trn, val, tst = build_data(o[:vggfile], o[:jsonfile])
    println("Data loaded...")

    # compile knet model
    if o[:loadfile] == nothing
        f = compile(:show_and_tell; out=o[:hidden], vocabsize=voc.size,embed=o[:embed])
    else
        f = load(o[:loadfile])
    end

    # set learning rate
    setp(f; lr=o[:lr])

    # training process
    println("Training has been started...")
    for epoch = 1:o[:epochs]
        train(f, trn, voc; gclip=o[:gclip])
        @printf("epoch:%d softloss:%g/%g\n", epoch,
                test(f, trn, voc),
                test(f, val, voc))
    end

    # save trained model
    o[:savefile]!=nothing && save(o[:savefile], "net", clean(net))
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
