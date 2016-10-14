using MAT, JSON, ArgParse, JLD
include("vocab.jl")


function main(args)
    s = ArgParseSettings()
    s.description = "Data preprocessing for Karpathy data."

    @add_arg_table s begin
        ("--vggfile"; help="MAT file contains VGG features")
        ("--jsonfile"; help="JSON file contains text data")
        ("--savefile"; help="data save file")
    end

    # parse args
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    # build vocabulary and split data
    voc, trn, val, tst = build_data(o[:vggfile], o[:jsonfile])
    println("Data processing completed.")
    flush(STDOUT)

    # save data
    save(o[:savefile], "voc", voc, "trn", trn, "val", val, "tst", tst)
    println("Processed data saved to ", o[:savefile])
    flush(STDOUT)
end


# for karpathy data
function build_data(vgg_filename, json_filename)
    vgg_fs = matread(vgg_filename)["feats"]
    json_data = JSON.parsefile(json_filename)
    images = json_data["images"]
    data, words = Dict(), Set()

    # data split, also build words set
    for i in 1:length(images)
        !haskey(data, images[i]["split"]) && push!(data, images[i]["split"] => [])
        push!(data[images[i]["split"]], (vgg_fs[:,i], images[i]))
    end

    # prepare words
    words = mapreduce(i -> mapreduce(s -> s["tokens"], vcat, i[2]["sentences"]),
                      vcat,
                      data["train"])

    # build vocabulary
    voc = Vocabulary(words)

    # build sentences
    helper(a) = mapreduce(e -> map(s -> (e[2]["filename"], e[1], sen2vec(voc, s["tokens"])), e[2]["sentences"]), vcat, a)
    trn, val, tst = map(i -> helper(data[i]), ["train", "val", "test"])

    return (voc, trn, val, tst)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
