using GPUChecker
using CUDArt
CUDArt.device(first_min_used_gpu())

using Knet
using ArgParse
using JLD

include("vocab.jl")
include("model.jl")

function main(args)
    s = ArgParseSettings()
    s.description = "Caption generation script for the model."

    @add_arg_table s begin
        ("--datafile"; help="data file contains vocabulary")
        ("--modelfile"; help="trained model file")
        ("--savedir"; help="save generations and references")
        ("--datasplit"; default="tst"; help="data split is going to be used")
        ("--maxlen"; arg_type=Int; default=20; help="max sentence length")
        ("--nogpu"; action=:store_true)
        ("--testing"; action=:store_true)
        ("--shuffle"; action=:store_true)
        ("--debug"; action=:store_true)
        ("--amount"; arg_type=Int; default=20; help="generation amount in case of testing")
    end

    println("Datetime: ", now()); flush(STDOUT)

    # parse args
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    # checkdir
    !isdir(o[:savedir]) && (println("savedir does not exist."); flush(STDOUT); quit())

    # load data
    tst = load(o[:datafile], o[:datasplit])
    voc = load(o[:datafile], "voc")
    net = load(o[:modelfile], "weights")
    @printf("Data loaded [%s]\n", now()); flush(STDOUT)

    o[:testing] && o[:shuffle] && shuffle!(tst)

    refs, gens = Dict(), Dict()
    counter, ti = 0, now()
    atype = o[:nogpu] ? Array{Float32} : KnetArray{Float32}

    # generate captions
    for i = 1:length(tst)
        o[:testing] && counter > o[:amount] && break
        gen = generate(atype, w, s, tst[i][2], voc, o[:maxlen])
        orig = vec2sen(voc, tst[i][3])

        !haskey(gens, fn) && (gens[fn] = gen; refs[fn] = Any[])
        push!(refs[fn], orig)

        if o[:debug]
            @printf("filename: %s\noriginal: %s\ngenerated: %s [%s]\n\n",
                    fn, orig, gen, now()); flush(STDOUT)
        end
        counter += 1
    end

    # some validation
    ks = sort(collect(keys(refs)))
    sz = length(refs[ks[1]])
    for i = 2:length(ks)
        length(refs[ks[i]]) == sz || (println("Validation error! ", ks[i], " ", sz); quit())
    end

    # open file streams
    files = Any[]
    for i = 1:sz
        f = open(abspath(joinpath(o[:savedir], "refs$(i).txt")), "w")
        println("file: ", f); flush(STDOUT)
        push!(files, f)
    end
    resfile = open(abspath(joinpath(o[:savedir], "results.txt")), "w")
    namefile = open(abspath(joinpath(o[:savedir], "filenames.txt")), "w")

    for i = 1:length(ks)
        write(resfile, string(gens[ks[i]], "\n"))
        write(namefile, string(ks[i], "\n"))
        for j = 1:sz
            write(files[j], string(refs[ks[i]][j], "\n"))
        end
    end

    # close file streams
    for i = 1:sz
        close(files[i])
    end
    close(resfile)
    close(namefile)

    tf = now()
    @printf("\nTime elapsed: %s [%s]\n", tf-ti, tf)
end


function generate(atype, w, s, vis, voc, maxlen)
    # feed visual features
    x = reshape(vis, 1, length(vis))
    x = convert(atype, x)
    x = x * w[5]
    (s[1], s[2]) = lstm(w[1], w[2], s[1], s[2], x)

    # language generation
    word = SOS
    sentence = Any[word]
    len = 1

    while word != EOS && len < maxlen
        x = reshape(word2onehot(voc, word), 1, voc.size)
        x = convert(atype, x) * w[6]
        (s[1], s[2]) = lstm(w[1], w[2], s[1], s[2], x);
        ypred = logp(s[1] * w[3] .+ w[4], 2)
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

vec2sen(voc::Vocabulary, vec) = join(map(i -> index2word(voc,i), vec[2:end-1]), " ")

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
