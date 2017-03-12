using Images
using JSON
using ArgParse
using HDF5

SPLITS = ["train", "restval", "val", "test"]

include("lib/data.jl")
include("lib/imgproc.jl")

function main(args)
    s = ArgParseSettings()
    s.description = "Convert common image captioning datasets to JLD format."

    @add_arg_table s begin
        ("--images"; required=true; help="images dir")
        ("--captions"; required=true; help="captions archive file (karpathy)")
        ("--savefile"; required=true; help="output file in HDF5 format")
        ("--imsize"; arg_type=Int; nargs=2; default=[224,224])
        ("--feedback"; arg_type=Int; default=0)
        ("--debug"; action=:store_true)
        ("--seed"; arg_type=Int; default=0; help="random seed")
        ("--nocrop"; action=:store_true)
        ("--randomcrop"; action=:store_true)
        ("--extradata"; action=:store_true)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); display(o)
    println(); flush(STDOUT)
    o[:seed] > 0 && srand(o[:seed])

    imgpath = abspath(o[:images])
    capfile = abspath(o[:captions])
    newsize = tuple(o[:imsize]...)
    crop = !o[:nocrop]
    savefile = abspath(o[:savefile])
    f = h5open(savefile, "w")

    # process images
    for splitname in SPLITS
        @printf("Processing %s split... [%s]\n", splitname, now())
        flush(STDOUT)

        randomcrop = false
        if splitname == "train" || (splitname == "restval" && o[:extradata])
            randomcrop = o[:randomcrop]
        end

        entries = first(get_entries(capfile, [splitname]))
        data = Any[]
        for i = 1:length(entries)
            entry = entries[i]
            if o[:debug]
                @printf("Image: %s\n", entry["filename"]); flush(STDOUT)
            end

            # processing
            img = read_image(entry["filename"], imgpath)
            img = process_image(img, newsize; crop=crop, randomcrop=randomcrop)

            write(f, entry["filename"], img)
            entries[i] = 0

            # feedback
            if o[:feedback] > 0 && i % o[:feedback] == 0
                @printf("Processed %d images by so far...\n", i)
                flush(STDOUT)
            end
        end
    end
    close(f)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
