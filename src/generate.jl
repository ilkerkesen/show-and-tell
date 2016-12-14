using Knet
using ArgParse
using JLD

include("vocab.jl")
include("model.jl")

function main(args)
    s = ArgParseSettings()
    s.description = "Caption generation script for the model."

    @add_arg_table s begin
        ("--images"; help="data file contains vocabulary")
        ("--captions"; help="data file contains vocabulary")
        ("--modelfile"; help="trained model file")
        ("--savedir"; help="save generations and references")
        ("--datasplit"; default="test"; help="data split is going to be used")
        ("--maxlen"; arg_type=Int; default=20; help="max sentence length")
        ("--nogpu"; action=:store_true)
        ("--testing"; action=:store_true)
        ("--shuffle"; action=:store_true)
        ("--debug"; action=:store_true)
        ("--amount"; arg_type=Int; default=20;
         help="generation amount in case of testing")
    end

    # parse args
    println("Datetime: ", now()); flush(STDOUT)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    # checkdir
    if !isdir(o[:savedir])
        println("savedir does not exist.")
        flush(STDOUT)
        quit()
    end

    # load data
    images = load(o[:images], o[:datasplit])
    captions = load(o[:captions], o[:datasplit])
    vocab = load(o[:captions], "vocab")
    @printf("Data loaded [%s]\n", now()); flush(STDOUT)

    # testing, I usually use this for train split
    o[:testing] && o[:shuffle] && shuffle!(tst)

    # load weights
    atype = o[:nogpu] ? Array{Float32} : KnetArray{Float32}
    w1 = load(o[:modelfile], "w1")
    w2 = load(o[:modelfile], "w2")
    lossval = load(o[:modelfile], "lossval")
    w1 = map(i->convert(atype, i), w1)
    w2 = map(i->convert(atype, i), w2)
    
    w = map(i->convert(atype, i), load(o[:modelfile], "weights"));
    s = initstate(atype, size(w[3], 1), 1)

    # generate captions
    counter, ti = 0, now()
    refs, gens = Dict(), Dict()
    @printf("Generation started (loss=%g,date=%s)\n", lossval, now())
    flush(STDOUT)
    for i = 1:length(images)
        o[:testing] && counter >= o[:amount] && break

        # check filenames
        f1, f2 = images[i][1], captions[i][1]
        f1 == f2 || error("filename mismatch")

        image = images[i][2]
        raw = captions[i][2]
        token = 
        gen = generate(atype, w, copy(s), tst[i][2], vocab, o[:maxlen])
        fn = tst[i][1]
        orig = vec2sen(vocab, tst[i][3])

        !haskey(gens, fn) && (gens[fn] = gen; refs[fn] = Any[])
        push!(refs[fn], orig)

        if o[:debug]
            @printf("filename: %s\noriginal: %s\ngenerated: %s [%s]\n\n",
                    fn, orig, gen, now()); flush(STDOUT)
        end
        counter += 1
    end

    o[:testing] && return
    
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

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
