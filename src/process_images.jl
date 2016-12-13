using ArgParse, JLD, Images, JSON

SPLITS = ["train", "restval", "val", "test"]


function main(args)
    s = ArgParseSettings()
    s.description = "Convert common image captioning datasets to JLD format."

    @add_arg_table s begin
        ("--images"; required=true; help="images archive file path")
        ("--captions"; required=true;
         help="captions archive file path (karpathy)")
        ("--savefile"; required=true; help="output file in JLD format")
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
            img = read_image(f, imgfile, tmpdir)
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


function get_filenames(zip, split)
    zip = abspath(zip)
    file = joinpath(splitext(splitdir(abspath(zip))[2])[1], "dataset.json")
    images = JSON.parse(readstring(`unzip -p $zip $file`))["images"]
    return map(j -> j["filename"], filter(i -> i["split"] == split, images))
end


function read_image(file, zip, tmp)
    source = joinpath("Flicker8k_Dataset", file)
    target = joinpath(tmp, file)
    img = readstring(`unzip -p $zip Flicker8k_Dataset/$file`)
    write(target, img)
    img = load(target)
    rm(target)
    return img
end


function process_image(img, newsize, rgbmean)
    a1 = Images.imresize(img, newsize) # resize image
    b1 = separate(a1) # separate image channels, build a tensor
    c1 = convert(Array{Float32}, b1) # type conversion
    d1 = reshape(c1[:,:,1:3], (newsize[1],newsize[2],3,1)) # reshape
    e1 = (255 * d1 .- rgbmean) # 8bit image representation
    return permutedims(e1, [2,1,3,4]) # transpose
end


!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
