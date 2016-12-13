function get_filenames(zip, split)
    zip = abspath(zip)
    file = joinpath(splitext(splitdir(abspath(zip))[2])[1], "dataset.json")
    images = JSON.parse(readstring(`unzip -p $zip $file`))["images"]
    return map(j -> j["filename"], filter(i -> i["split"] == split, images))
end


function process_image(img, newsize, rgbmean)
    a1 = Images.imresize(img, newsize) # resize image
    b1 = separate(a1) # separate image channels, build a tensor
    c1 = convert(Array{Float32}, b1) # type conversion
    d1 = reshape(c1[:,:,1:3], (newsize[1],newsize[2],3,1)) # reshape
    e1 = (255 * d1 .- rgbmean) # 8bit image representation
    return permutedims(e1, [2,1,3,4]) # transpose
end
