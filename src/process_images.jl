using ArgParse, JLD, Images, JSON


function main(args)
    s = ArgParseSettings()
    s.description = "Convert common image captioning datasets to JLD format."

    @add_arg_table s begin
        ("--images"; required=true; help="images archive file path")
        ("--captions"; required=true; help="captions archive file path (karpathy)")
        ("--savefile"; required=true; help="output file in JLD format")
        ("--tmpdir"; default=abspath("~/tmp"); help="tmpdir for processing")
        ("--imsize"; arg_type=Int64; nargs=2; default=[224,224];
         help="new image sizes")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    imgfile = abspath(o[:images])
    capfile = abspath(o[:captions])
    splits = get_filenames(capfile) # trn, val, tst
    tmpdir = abspath(o[:tmpdir])
    imsize = tuple(o[:imsize]...)

    # prepare tmpdir 
    !isdir(tmpdir) && mkpath(tmpdir)

    # process images
    trn, val, tst = map(
        s -> map(i -> process_image(read_image(i, imgfile, tmpdir),
                                        imsize), s), splits)

    # compose filenames and images and save processed data
    trn = Dict("filenames" => splits[1], "images" => trn)
    val = Dict("filenames" => splits[2], "images" => val)
    tst = Dict("filenames" => splits[3], "images" => tst)
    save(o[:savefile], "trn", trn, "val", val, "tst", tst)
end


function get_filenames(zip)
    zip = abspath(zip)
    file = joinpath(split(splitdir(abspath(zip))[2], ".")[1], "dataset.json")
    images = JSON.parse(readstring(`unzip -p $zip $file`))["images"]
    map(s -> map(j -> j["filename"], filter(i -> i["split"] == s, images)),
        ["train", "val", "test"])
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


function process_image(img, newsize)
    a1 = Images.imresize(img, newsize) # resize image
    b1 = separate(a1) # separate image channels, build a tensor
    c1 = convert(Array{Float32}, b1) # type conversion
    d1 = reshape(c1[:,:,1:3], (newsize[1],newsize[2],3,1)) # reshape
    e1 = (255 * d1) # 8bit image representation
    return permutedims(e1, [2,1,3,4]) # transpose
end


!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
