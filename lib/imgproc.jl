function read_image(file, imgpath)
    target = joinpath(imgpath, file)
    img = load(target)
    return img
end

function process_image(
    img, newsize; rgbmean=nothing, crop=true, randomcrop=false)
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

    b1 = permutedims(channelview(a1), (3,2,1))
    colordim = size(b1, 3)
    if colordim != 3
        c1 = convert(Array{Float32}, b1)
        c1 = cat(3, cat(3, c1, c1), c1)
    else
        c1 = convert(Array{Float32}, b1) # type conversion
    end
    d1 = reshape(c1[:,:,1:3], (newsize[1],newsize[2],3,1)) # reshape
    e1 = (255 * d1) # 8bit image representation
    if rgbmean != nothing
        e1 = e1 .- rgbmean
    end

    return permutedims(e1, [2,1,3,4]) # transpose
end
