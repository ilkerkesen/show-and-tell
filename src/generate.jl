using Knet
using ArgParse
using JLD

include("vocab.jl")
include("model.jl")

function main(args)
    s = ArgParseSettings()
    s.description = "Caption generation script for the model."

    @add_arg_table s begin
        ("--datafile"; help="data file contains dataset splits and vocabulary")
        ("--loadfile"; help="pretrained model file")
        ("--savefile"; help="save generated captions")
        ("--maxlen"; arg_type=Int; default=20; help="max sentence length")
    end

    # parse args
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    # load data
    tst = load(o[:datafile], "tst")
    voc = load(o[:datafile], "voc")
    net = load(o[:loadfile], "net")
    println("Data loaded...")
    flush(STDOUT)

    filenames = Any[]
    captions = Any[]
    for i = 1:length(tst)
        fn, cap = generate(net, tst[i], voc, o[:maxlen])
        if !(fn in filenames)
            push!(filenames, fn)
            push!(captions, cap)
            println(fn, ": ", cap)
            flush(STDOUT)
        end
    end

    # FIXME: make data convenient for perl script
    save(o[:savefile], "filenames", filenames)
    save(o[:savefile], "captions", captions)
end

function generate(f, sample, voc, maxlen)
    reset!(f)
    fn, vis, _ = sample
    vis = reshape(vis, length(vis), 1)
    forw(f, vis; decoding=false)
    word = SOS
    sentence = Any[word]

    len = 1
    while word != EOS && len < maxlen
        onehot = reshape(word2onehot(voc, word), voc.size, 1)
        ypred = forw(f, onehot; decoding=true)
        ypred = convert(Array{Float32}, ypred)
        word = index2word(voc, indmax(ypred))
        push!(sentence, word)
        len += 1
    end

    if word == EOS
        pop!(sentence)
    end

    return (fn, join(sentence[2:end], " "))
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
