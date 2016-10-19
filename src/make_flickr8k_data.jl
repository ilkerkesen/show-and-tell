using ArgParse, JLD, Images


function main(args)
    s = ArgParseSettings()
    s.description = "Convert Flickr datasets to JLD format."

    @add_arg_table s begin
        ("--readfiles"; required=true; nargs=2; help="dataset input files")
        ("--writefile"; required=true; help="output file in JLD")
        ("--tmpdir"; required=true; help="tmp dir for file processing")
        ("--width"; default=224; help="image width for resize operation")
        ("--means"; nargs=3; default=[123.68, 116.779, 103.939];
         help="RGB mean values for normalization")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    datafile, textfile = map(abspath, o[:readfiles])
    splits = get_filenames(textfile) # trn, val, tst
    tmpdir = abspath(o[:tmpdir])

    # prepare tmpdir 
    !isdir(o[:tmpdir]) && mkpath(o[:tmpdir])

    # process images
    trnimgs, valimgs, tstimgs = map(
        s -> map(i -> process_image(read_image(i, datafile, tmpdir), o[:width], o[:means]), s),
        splits)

    # save processed images
    save(o[:writefile], "trnimgs", trnimgs, "valimgs", valimgs, "tstimgs", tstimgs)
end


function get_filenames(file)
    split(readstring(`unzip -p $file Flickr_8k.trainImages.txt`), "\n")[1:end-1],
    split(readstring(`unzip -p $file Flickr_8k.devImages.txt`), "\n")[1:end-1],
    split(readstring(`unzip -p $file Flickr_8k.testImages.txt`), "\n")[1:end-1]
end


function read_image(file, zip, tmp)
    source = joinpath("Flicker8k_Dataset", file)
    target = joinpath(tmp, file)
    img = readstring(`unzip -p $zip Flicker8k_Dataset/$file`)
    write(target, img)
    img = load(abspath(joinpath(tmp, file)))
    rm(target)
    return img
end


function process_image(img, width, means)
    means = reshape(means, 1, 1, 3)
    mn = ntuple(i->div(size(img,i)*width,minimum(size(img))),2) # new size
    a1 = Images.imresize(img, mn) # resize image
    i1 = div(size(a1,1)-width,2)
    j1 = div(size(a1,2)-width,2)
    b1 = a1[i1+1:i1+224,j1+1:j1+224] # center cropping
    c1 = separate(b1) # separate image channels, build a tensor
    d1 = convert(Array{Float32}, c1) # type conversion
    e1 = reshape(d1[:,:,1:3], (width,width,3,1)) # reshape
    f1 = (255 * e1 .- means) # 8bit representation+normalization
    return permutedims(f1, [2,1,3,4]) # transpose
end


!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
