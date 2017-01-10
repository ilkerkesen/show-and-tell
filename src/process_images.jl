using ArgParse, JLD, Images, JSON
SPLITS = ["train", "restval", "val", "test"]

function main(args)
    s = ArgParseSettings()
    s.description = "Convert common image captioning datasets to JLD format."

    @add_arg_table s begin
        ("--images"; required=true; help="images dir")
        ("--captions"; required=true;
         help="captions archive file path (karpathy)")
        ("--savefile"; required=true; help="output file in JLD format")
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

    # process images
    data = Dict()
    for splitname in SPLITS
        @printf("Processing %s split... [%s]\n", splitname, now())
        flush(STDOUT)

        randomcrop = false
        if splitname == "train" || (splitname == "restval" && o[:extradata])
            randomcrop = o[:randomcrop]
        end

        filenames = get_filenames(capfile, splitname)
        splitdata = []
        counter = 0
        for f in filenames
            if o[:debug]
                @printf("Image: %s\n", f); flush(STDOUT)
            end

            # processing
            img = read_image(f, imgpath)
            img = process_image(
                img, newsize, rgbmean; crop=crop, randomcrop=randomcrop)
            push!(splitdata, (f, img))

            # feedback
            counter += 1
            if o[:feedback] > 0 && counter % o[:feedback] == 0
                @printf("Processed %d images by so far...\n", counter)
                flush(STDOUT)
            end
        end
        data[splitname] = splitdata
    end

    # save images
    save(o[:savefile],
         "train", data["train"],
         "restval", data["restval"],
         "val", data["val"],
         "test", data["test"])
    @printf("Processed images saved to %s. [%s]\n", o[:savefile], now())
end

function read_image(file, imgpath)
    target = joinpath(imgpath, file)
    img = load(target)
    return img
end

function get_filenames(zip, split)
    zip = abspath(zip)
    file = joinpath(splitext(splitdir(abspath(zip))[2])[1], "dataset.json")
    images = JSON.parse(readstring(`unzip -p $zip $file`))["images"]
    return map(j -> j["filename"], filter(i -> i["split"] == split, images))
end

function process_image(img, newsize, rgbmean; crop=true, randomcrop=false)
    scaled = ntuple(i->div(size(img,i)*newsize[i],minimum(size(img))),2)
    a1 = Images.imresize(img, scaled)

    # randomcrop vs. centercrop
    if randomcrop
        offsets = ntuple(i->rand(1:scaled[i]-minimum(scaled)+1),2)
    else
        offsets = ntuple(i->div(size(a1,i)-newsize[i],2)+1,2)
    end

    if crop
        a1 = a1[offsets[1]:offsets[1]+newsize[1]-1,
                offsets[2]:offsets[2]+newsize[2]-1]
    else
        a1 = Images.imresize(a1, newsize)
    end

    b1 = separate(a1) # separate image channels, build a tensor
    colordim = size(b1, 3)
    colorspace = img.properties["colorspace"]
    if colordim != 3 || colorspace == "Gray"
        c1 = convert(Array{Float32}, b1.data)
        c1 = cat(3, cat(3, c1, c1), c1)
    else
        c1 = convert(Array{Float32}, b1) # type conversion
    end
    d1 = reshape(c1[:,:,1:3], (newsize[1],newsize[2],3,1)) # reshape
    e1 = (255 * d1 .- rgbmean) # 8bit image representation
    return permutedims(e1, [2,1,3,4]) # transpose
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
