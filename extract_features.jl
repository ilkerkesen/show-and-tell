using Knet, ArgParse, JLD, MAT
include("lib/convnet.jl")
SPLITS = ["train", "restval", "val", "test"]

function main(args)
    s = ArgParseSettings()
    s.description = "Extract CNN features of images (now only just for VGG-19)"

    @add_arg_table s begin
        ("--datafile"; help="data file in JLD format")
        ("--savefile"; help="save file in JLD format")
        ("--cnnfile"; help="CNN model file")
        ("--convnet"; default="vgg19")
        ("--lastlayer"; default="relu7"; help="layer for feature extraction")
        ("--seed"; arg_type=Int; default=1; help="random seed")
        ("--fc6drop"; arg_type=Float32; default=Float32(0.0))
        ("--feedback"; arg_type=Int; default=0; help="feedback in every N image")
        ("--extradata"; action=:store_true)
    end

    # parse args
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:seed] > 0 && srand(o[:seed])

    # load data
    @printf("Data loading...\n"); flush(STDOUT)
    data = load(o[:datafile], "data")
    splitname = data[1]["split"]

    # load CNN
    if !startswith(o[:convnet], "vgg")
        error("only VGG models are supported")
    end
    @printf("CNN loading...\n"); flush(STDOUT)
    lastlayer = o[:lastlayer]
    convnet = eval(parse(o[:convnet]))
    CNNmat = matread(o[:cnnfile])
    weights = get_vgg_weights(CNNmat; last_layer=lastlayer)
    @printf("Done.\n"); flush(STDOUT)

    # feature extraction
    @printf("Feature extraction for %s split...\n", splitname)
    flush(STDOUT)
    dropouts = Dict()
    if splitname == "train" || (splitname == "restval" && o[:extradata])
        dropouts = Dict("fc6drop" => o[:fc6drop])
    end
    for i = 1:length(data)
        image = data[i]["image"]
        feats = convnet(weights, KnetArray(image); dropouts=dropouts)
        feats = convert(Array{Float32}, feats)
        feats = reshape(feats, 1, length(feats))
        data[i]["image"] = feats
        if o[:feedback] > 0 && i % o[:feedback] == 0
            @printf("[%%%.1f] %d images processed so far [%s]\n",
                    round(100*i/length(data), 1), i, now())
            flush(STDOUT)
        end
    end

    # save features
    @printf("Save extracted features to output file... "); flush(STDOUT)
    save(o[:savefile], "data", data)
    @printf("Totally done.\n"); flush(STDOUT)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
