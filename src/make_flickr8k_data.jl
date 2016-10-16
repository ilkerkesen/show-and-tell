using ArgParse, JLD, Images

function main(args)
    s = ArgParseSettings()
    s.description = "Convert Flickr datasets to JLD format."

    @add_arg_table s begin
        ("--readfiles"; required=true; nargs=2; help="dataset input files")
        ("--writefile"; required=true; help="output file in JLD")
        ("--tmpdir"; required=true; help="tmp dir for file processing")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    datafile, textfile = map(abspath, o[:readfiles])
    trnfiles, valfiles, tstfiles = get_filenames(textfile)
    tmpdir = abspath(o[:tmpdir])

    # prepare tmpdir
    !isdir(o[:tmpdir]) && mkpath(o[:tmpdir])
end

function get_filenames(file)
    split(readstring(`unzip -p $file Flickr_8k.trainImages.txt`), "\n")[1:end-1],
    split(readstring(`unzip -p $file Flickr_8k.devImages.txt`), "\n")[1:end-1],
    split(readstring(`unzip -p $file Flickr_8k.testImages.txt`), "\n")[1:end-1]
end

function read_image(file, zip, tmp)
    readstring(`unzip -p $zip Flicker8k_Dataset/$file > $tmp`)
    img = load(abspath(joinpath(tmp, file)))
    rm(abspath(joinpath(tmp, file)))
    return img
end

function process_image(img; width=299)
    mn = ntuple(i->div(size(img,i)*width,minimum(size(img))),2) # new size
    a1 = Images.imresize(a0, new_size) # resize image
    b1 = separate(a1) # separate image channels, build a tensor
    c1 = convert(Array{Float32}, b1) # type conversion
    d1 = reshape(c1[:,:,1:3], (width,width,3,1)) # reshape
    e1 = (255 * d1) # 8bit image representation
    f1 = permutedims(e1, [2,1,3,4]) # transpose
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
