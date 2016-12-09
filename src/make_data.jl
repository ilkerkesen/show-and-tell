using ArgParse, JLD, JSON

include("vocab.jl");

function main(args=ARGS)
    s = ArgParseSettings()
    s.description = string(
        "Show and Tell: A Neural Image Caption Generator",
        " Knet implementation by Ilker Kesen [ikesen16_at_ku.edu.tr], 2016.")

    @add_arg_table s begin
        ("--images"; help="images file")
        ("--captions"; help="captions file")
        ("--savefile"; help="output file")
    end

    # parse args
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    # build vocabulary and split data
    voc, trn, val, tst = build_data(o[:images], o[:captions])
    println("Data preprocessing completed."); flush(STDOUT)

    # save data
    save(o[:savefile], "voc", voc, "trn", trn, "val", val, "tst", tst)
    println("Processed data saved to ", o[:savefile]); flush(STDOUT)
end


function build_data(imagesfile, zip)
    zip = abspath(zip)
    file = joinpath(split(splitdir(abspath(zip))[2], ".")[1], "dataset.json")
    samples = JSON.parse(readstring(`unzip -p $zip $file`))["images"]
    imgdata = load(abspath(imagesfile))
    data, words = Dict("trn" => [], "val" => [], "tst" => []), Set()
    stats = Dict()

    # build some statistics
    for sample in samples
        if
    end

    splits = keys(feats)
    s, i, j = 1, 1, 1
    while true
        push!(data[s], (feats, ))
    end
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
