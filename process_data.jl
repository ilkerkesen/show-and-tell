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
        ("--captions"; required=true;
         help="captions archive file path (karpathy)")
        ("--savefile"; required=true; help="output file in HDF5 format")
        ("--imsize"; arg_type=Int; nargs=2; default=[224,224];
         help="new image sizes")
        ("--rgbmean"; arg_type=Float32; nargs=3;
         default=map(Float32, [123.68, 116.779, 103.939]))
        ("--feedback"; arg_type=Int; default=0;
         help="period of displaying number of images processed")
        ("--debug"; action=:store_true)
        ("--seed"; arg_type=Int; default=1; help="random seed")
        ("--nocrop"; action=:store_true)
        ("--randomcrop"; action=:store_true)
        ("--extradata"; action=:store_true)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:seed] > 0 && srand(o[:seed])

    imgpath = abspath(o[:images])
    capfile = abspath(o[:captions])
    newsize = tuple(o[:imsize]...)
    rgbmean = reshape(o[:rgbmean], (1,1,3))
    crop = !o[:nocrop]
    savefile, ext = splitext(abspath(o[:savefile]))
    ext == "" || ext == ".jld" || error("invalid extension")

    # process images
    for splitname in SPLITS
        @printf("Processing %s split... [%s]\n", splitname, now())
        flush(STDOUT)

        randomcrop = false
        if splitname == "train" || (splitname == "restval" && o[:extradata])
            randomcrop = o[:randomcrop]
        end

        entries = get_entries(capfile, splitname)
        data = Any[]
        for i = 1:length(entries)
            entry = entries[i]
            if o[:debug]
                @printf("Image: %s\n", entry["filename"]); flush(STDOUT)
            end

            # processing
            img = read_image(entry["filename"], imgpath)
            img = process_image(
                img, newsize, rgbmean; crop=crop, randomcrop=randomcrop)
            entry["image"] = img
            push!(data, entry)
            entries[i] = 0

            # feedback
            if o[:feedback] > 0 && i % o[:feedback] == 0
                @printf("Processed %d images by so far...\n", i)
                flush(STDOUT)
            end

            # save data to file
            if i % partsize == 0 || i == length(entries)
                d,r = divrem(i, partsize)
                partnumber = d + (r != 0)
                filename = @sprintf(
                    "%s-%s-part-%02d.jld", savefile, splitname, partnumber)
                save(filename, "data", data)
                empty!(data)
                gc()
                @printf("(%s,%d) saved to %s.\n",
                        splitname, partnumber, filename)
            end
        end
    end
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
