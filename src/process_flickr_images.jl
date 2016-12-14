using ArgParse, JLD, Images, JSON
SPLITS = ["train", "restval", "val", "test"]
include("imgproc.jl")
include("util.jl")

function main(args)
    s = ArgParseSettings()
    s.description = "Convert common image captioning datasets to JLD format."

    @add_arg_table s begin
        ("--images"; required=true; help="images archive file path")
        ("--captions"; required=true;
         help="captions archive file path (karpathy)")
        ("--savefile"; required=true; help="output file in JLD format")
        ("--dataset"; default="flickr8k";
         help="dataset name (flickr8k|flickr30k)")
        ("--tmpdir"; default=joinpath(homedir(), "tmp");
         help="tmpdir for processing")
        ("--imsize"; arg_type=Int; nargs=2; default=[224,224];
         help="new image sizes")
        ("--rgbmean"; arg_type=Float32; nargs=3;
         default=map(Float32, [123.68, 116.779, 103.939]))
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    imgfile = abspath(o[:images])
    capfile = abspath(o[:captions])
    tmpdir = abspath(o[:tmpdir])
    newsize = tuple(o[:imsize]...)
    rgbmean = reshape(o[:rgbmean], (1,1,3))

    # prepare tmpdir 
    !isdir(tmpdir) && mkpath(tmpdir)

    # process images
    data = Dict()
    for splitname in SPLITS
        @printf("Processing %s split...\n", splitname)
        filenames = get_filenames(capfile, splitname)
        splitdata = []
        for f in filenames
            img = read_image(f, imgfile, tmpdir, o[:dataset])
            img = process_image(img, newsize, rgbmean)
            push!(splitdata, (f, img))
        end
        data[splitname] = splitdata
    end

    # save images
    save(o[:savefile],
         "train", data["train"],
         "restval", data["restval"],
         "val", data["val"],
         "test", data["test"])
end

function read_image(file, zip, tmp, dataset)
    prefix = "Flicker8k_Dataset"
    if dataset == "flickr30k"
        prefix = "flickr30k-images"
    end

    source = joinpath(prefix, file)
    target = joinpath(tmp, file)
    if dataset == "flickr30k"
        extract_file_from_tar(zip, source, tmp)
    else
        extract_file_from_zip(zip, source, target)
    end

    img = load(target)
    rm(target)
    return img
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
