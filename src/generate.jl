using Knet
using ArgParse
using JLD

include("vocab.jl")
include("convnet.jl")
include("model.jl")

function main(args)
    s = ArgParseSettings()
    s.description = "Caption generation script for the model."

    @add_arg_table s begin
        ("--visual"; help="data file contains vocabulary")
        ("--captions"; help="data file contains vocabulary")
        ("--modelfile"; help="trained model file")
        ("--cnnfile"; help="convnet file for non-finetuned model")
        ("--savedir"; help="save generations and references")
        ("--beamsize"; arg_type=Int; default=1)
        ("--datasplit"; default="test"; help="data split is going to be used")
        ("--maxlen"; arg_type=Int; default=20; help="max sentence length")
        ("--nogpu"; action=:store_true)
        ("--testing"; action=:store_true)
        ("--shuffle"; action=:store_true)
        ("--debug"; action=:store_true)
        ("--amount"; arg_type=Int; default=20;
         help="generation amount in case of testing")
        ("--noreferences"; action=:store_true;
         help="this is for non-dataset images")
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
    savedir = abspath(o[:savedir])

    # load data
    visuals = load(o[:visual], o[:datasplit])
    captions = load(o[:captions], o[:datasplit])
    vocab = load(o[:captions], "vocab")
    @printf("Data loaded [%s]\n", now()); flush(STDOUT)

    # load weights
    atype = o[:nogpu] ? Array{Float32} : KnetArray{Float32}
    w = load(o[:modelfile], "w")
    w = map(i->convert(atype, i), w)
    lossval = load(o[:modelfile], "lossval")
    s = initstate(atype, size(w[3], 1), 1)

    wcnn = load(o[:modelfile], "wcnn")
    if wcnn != nothing
        wcnn = map(i->convert(atype, i), wcnn)
    end

    # generate captions
    counter, ti = 0, now()
    filenames, references, generations = [], [], []
    @printf("Generation started (loss=%g,date=%s)\n", lossval, now())
    flush(STDOUT)
    for i = 1:length(visuals)
        o[:testing] && counter >= o[:amount] && break
        filename1, filename2 = visuals[i][1], captions[i][1]
        filename1 == filename2 || error("filename mismatch")
        visual = visuals[i][2]
        sentences = map(s -> s[1], captions[i][2])
        generated = generate(
            w, wcnn, copy(s), visual, vocab, o[:maxlen]; beamsize=o[:beamsize])
        o[:debug] && report_generation(filename1, generated, sentences)
        push!(filenames, filename1)
        push!(references, sentences)
        push!(generations, generated)
        counter += 1
    end

    if !o[:testing]
        write_generations(savedir, filenames, generations, references)
    end

    tf = now()
    @printf("\nTime elapsed: %s [%s]\n", tf-ti, tf)
end

function report_generation(filename, generated, references)
    @printf("\nFilename: %s\n", filename)
    @printf("Generated: %s\n", generated)
    for i = 1:length(references)
        @printf("Reference #%d: %s\n", i, references[i])
    end
    flush(STDOUT)
end

function write_generations(savedir, filenames, generations, references)
    numrefs = length(references[1])
    files = []
    for i = 1:numrefs
        file = open(joinpath(savedir, "references$(i).txt"), "w")
        push!(files, file)
    end
    filenamesfile = open(joinpath(savedir, "filenames.txt"), "w")
    generationsfile = open(joinpath(savedir, "generations.txt"), "w")
    for i = 1:length(filenames)
        write(filenamesfile, string(filenames[i], "\n"))
        write(generationsfile, string(generations[i], "\n"))
        for j = 1:numrefs
            write(files[j], string(references[i][j], "\n"))
        end
    end
    for i = 1:numrefs
        close(files[i])
    end
    close(filenamesfile)
    close(generationsfile)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
