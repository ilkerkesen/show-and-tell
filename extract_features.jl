using Knet, ArgParse, HDF5, MAT, JSON
include("lib/convnet.jl")
include("lib/imgproc.jl")
include("lib/data.jl")
SPLITS = ["train", "restval", "val", "test"]

function main(args)
    s = ArgParseSettings()
    s.description = "Extract CNN features of images (now only just for VGG-19)"

    @add_arg_table s begin
        ("--images"; help="images file in JLD format")
        ("--captions"; help="captions archieve (karpathy)")
        ("--savefile"; help="save file in JLD format")
        ("--cnnfile"; help="CNN model file")
        ("--convnet"; default="vgg16")
        ("--lastlayer"; default="relu7"; help="layer for feature extraction")
        ("--seed"; arg_type=Int; default=0; help="random seed")
        ("--fc6drop"; arg_type=Float32; default=Float32(0.0))
        ("--feedback"; arg_type=Int; default=0; help="feedback in every N image")
        ("--extradata"; action=:store_true)
    end

    # parse args
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:seed] > 0 && srand(o[:seed])

    # load CNN
    @printf("CNN loading...\n"); flush(STDOUT)
    lastlayer = o[:lastlayer]
    convnet = eval(parse(o[:convnet]))
    CNNmat = matread(o[:cnnfile])
    weights = get_vgg_weights(CNNmat; last_layer=lastlayer)
    avgimg = CNNmat["meta"]["normalization"]["averageImage"]
    avgimg = KnetArray(make_average_image(avgimg))

    # FIXME: this only works for vgg19 right now
    featuremaps = lastlayer == "conv5_4"
    if featuremaps
        @printf("featuremaps from %s layer.\n", lastlayer)
    end
    @printf("Done.\n"); flush(STDOUT)

    # feature extraction
    h5open(abspath(o[:savefile]), "w") do outfile
        entries = get_entries(o[:captions], SPLITS)
        for (k,splitname) in enumerate(SPLITS)
            @printf("Feature extraction for %s split...\n", splitname)
            flush(STDOUT)
            dropouts = Dict()
            if splitname == "train" || (splitname == "restval" && o[:extradata])
                dropouts = Dict(:fc6drop => o[:fc6drop])
            end
            for i = 1:length(entries[k])
                filename = entries[k][i]["filename"]
                image = h5open(o[:images], "r") do f
                    read(f, filename)
                end
                dict = Dict(:featuremaps => featuremaps)
                feats = convnet(weights, KnetArray(image) .- avgimg; o=dict)
                feats = convert(Array{Float32}, feats)
                if !featuremaps
                    feats = reshape(feats, length(feats), 1)
                end
                write(outfile, filename, feats)

                if o[:feedback] > 0 && i % o[:feedback] == 0
                    @printf("[%%%.1f] %d images processed so far [%s]\n",
                            round(100*i/length(entries[k]), 1), i, now())
                    flush(STDOUT)
                end
            end
        end
    end

    @printf("Save extracted features to %s\n", abspath(o[:savefile]))
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
