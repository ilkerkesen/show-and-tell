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
        ("--datasplit"; default="tst"; help="data split is going to be used")
        ("--maxlen"; arg_type=Int; default=20; help="max sentence length")
        ("--testing"; action=:store_true)
        ("--amount"; arg_type=Int; default=20; help="generation amount in case of testing")
    end

    println("Datetime: ", now()); flush(STDOUT)

    # parse args
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    # load data
    tst = load(o[:datafile], o[:datasplit])
    voc = load(o[:datafile], "voc")
    net = load(o[:loadfile], "net")
    @printf("Data loaded [%s]\n", now()); flush(STDOUT)

    filenames = Any[]
    captions = Any[]
    counter, ti = 0, now()
    for i = 1:length(tst)
        o[:testing] && counter > o[:amount] && break
        fn, orig, gen = generate(net, tst[i], voc, o[:maxlen])

        if !(fn in filenames)
            push!(filenames, fn)
            push!(captions, gen)
            @printf("filename: %s\noriginal: %s\ngenerated: %s\n\n",
                    fn, orig, gen); flush(STDOUT)
            counter += 1
        end
    end

    tf = now()
    @printf("\nTime elapsed: %s [%s]\n", tf-ti, tf)

    # FIXME: make data convenient for perl script
    !testing && save(o[:savefile], "filenames", filenames, "captions", captions)
end

function generate(f, sample, voc, maxlen)
    reset!(f)
    fn, vis, vec = sample
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

    orig = join(map(i -> index2word(voc,i), vec[2:end-1]), " ")
    gen = join(sentence[2:end], " ")
    return (fn, orig, gen)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
