using Knet
using ArgParse
using JLD
using HDF5
using JSON

include("lib/vocab.jl")
include("lib/base.jl")
include("lib/convnet.jl")
include("lib/init.jl")
include("lib/model.jl")
include("lib/data.jl")

function main(args)
    s = ArgParseSettings()
    s.description = "Caption generation script for the model."

    @add_arg_table s begin
        ("--images"; help="data file contains images/features")
        ("--captions"; help="zip file contains caption data (karpathy)")
        ("--modelfile"; help="trained model file")
        ("--vocabfile"; help="vocabulary file")
        ("--cnnfile"; help="convnet file for non-finetuned model")
        ("--savefile"; help="save generations in JSON format")
        ("--beamsize"; arg_type=Int; default=1)
        ("--datasplit"; default="test"; help="data split is going to be used")
        ("--extradata"; action=:store_true)
        ("--maxlen"; arg_type=Int; default=25; help="max sentence length")
        ("--nogpu"; action=:store_true)
        ("--testing"; action=:store_true)
        ("--shuffle"; action=:store_true)
        ("--amount"; arg_type=Int; default=20;
         help="generation amount in case of testing")
        ("--feedback"; arg_type=Int; default=100;
         help="feedback in every N generation")
    end

    # parse args
    println("Datetime: ", now()); flush(STDOUT)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    savefile = abspath(o[:savefile])

    # load data
    vocab = load(o[:vocabfile], "vocab")
    entries = get_entries(o[:captions], [o[:datasplit]])[1]
    @printf("Data loaded [%s]\n", now()); flush(STDOUT)

    # load weights
    atype = o[:nogpu] ? Array{Float32} : KnetArray{Float32}
    w = load(o[:modelfile], "w")
    w = map(i->convert(atype, i), w)
    score = load(o[:modelfile], "bestscore")
    s = initstate(atype, size(w[3], 1), 1)

    wcnn = load(o[:modelfile], "wcnn")
    if wcnn != nothing
        wcnn = map(i->convert(atype, i), wcnn)
    end

    @printf("Model loaded [%s]\n", now()); flush(STDOUT)

    # generate captions
    ti = now()
    captions = []
    @printf("Generation started (score=%g,date=%s)\n", score, now())
    flush(STDOUT)
    for i = 1:length(entries)
        o[:testing] && counter >= o[:amount] && break

        entry = entries[i]
        filename = entry["filename"]
        image = h5open(o[:images], "r") do f
            read(f, filename)
        end
        caption = generate(w, wcnn, copy(s), image, vocab;
                           maxlen=o[:maxlen], beamsize=o[:beamsize])
        references = map(s->s["raw"], entry["sentences"])
        new_entry = Dict(
            "filename" => filename,
            "hypothesis" => caption,
            "references" => references)
        push!(captions, new_entry)

        if o[:feedback] > 0 && i % o[:feedback] == 0
            @printf("\n%d captions generated so far [%s]\n", i, now())
            flush(STDOUT)
        end
    end

    meta = Dict(
        "model" => o[:modelfile],
        "data" => o[:images],
        "split" => o[:datasplit],
        "score" => score,
        "beamsize" => o[:beamsize],
        "date" => now())

    open(savefile, "w") do f
        write(f, json(Dict("meta"=>meta,"captions"=>captions)))
    end

    tf = now()
    @printf("\nTime elapsed: %s [%s]\n", tf-ti, tf)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
