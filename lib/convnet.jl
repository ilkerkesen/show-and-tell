# VGG16 model for convolutional feature extraction
function vgg16(w, x; o=Dict())
    # get parameters
    fc6drop = get(o, :fc6drop, 0.0)
    fc7drop = get(o, :fc7drop, 0.0)
    mode = get(o, :cnnmode, 1)
    featuremaps = get(o, :featuremaps, false)

    conv1_1 = conv4(w[1], x; padding=1, mode=mode) .+ w[2]
    conv1_1 = relu(conv1_1)
    conv1_2 = conv4(w[3], conv1_1; padding=1, mode=mode) .+ w[4]
    conv1_2 = relu(conv1_2)
    pool1   = pool(conv1_2)

    conv2_1 = conv4(w[5], pool1; padding=1, mode=mode) .+ w[6]
    conv2_1 = relu(conv2_1)
    conv2_2 = conv4(w[7], conv2_1; padding=1, mode=mode) .+ w[8]
    conv2_2 = relu(conv2_2)
    pool2   = pool(conv2_2)

    conv3_1 = conv4(w[9], pool2; padding=1, mode=mode) .+ w[10]
    conv3_1 = relu(conv3_1)
    conv3_2 = conv4(w[11], conv3_1; padding=1, mode=mode) .+ w[12]
    conv3_2 = relu(conv3_2)
    conv3_3 = conv4(w[13], conv3_2; padding=1, mode=mode) .+ w[14]
    conv3_3 = relu(conv3_3)
    pool3   = pool(conv3_3)

    conv4_1 = conv4(w[15], pool3; padding=1, mode=mode) .+ w[16]
    conv4_1 = relu(conv4_1)
    conv4_2 = conv4(w[17], conv4_1; padding=1, mode=mode) .+ w[18]
    conv4_2 = relu(conv4_2)
    conv4_3 = conv4(w[19], conv4_2; padding=1, mode=mode) .+ w[20]
    conv4_3 = relu(conv4_3)
    pool4   = pool(conv4_3)

    conv5_1 = conv4(w[21], pool4; padding=1, mode=mode) .+ w[22]
    conv5_1 = relu(conv5_1)
    conv5_2 = conv4(w[23], conv5_1; padding=1, mode=mode) .+ w[24]
    conv5_2 = relu(conv5_2)
    conv5_3 = conv4(w[25], conv5_2; padding=1, mode=mode) .+ w[26]
    conv5_3 = relu(conv5_3)

    if !featuremaps
        pool5 = pool(conv5_3)
        fc6 = w[27] * mat(pool5) .+ w[28]
        fc6 = relu(fc6)
        fc6 = dropout(fc6, fc6drop)
        fc7 = w[29] * mat(fc6) .+ w[30]
        fc7 = relu(fc7)
    end
end

# VGG16 model for convolutional feature extraction
function vgg19(w, x; o=Dict())
    # get parameters
    fc6drop = get(o, :fc6drop, 0.0)
    fc7drop = get(o, :fc7drop, 0.0)
    mode = get(o, :cnnmode, 1)
    featuremaps = get(o, :featuremaps, false)

    conv1_1 = conv4(w[1], x; padding=1, mode=mode) .+ w[2]
    conv1_1 = relu(conv1_1)
    conv1_2 = conv4(w[3], conv1_1; padding=1, mode=mode) .+ w[4]
    conv1_2 = relu(conv1_2)
    pool1   = pool(conv1_2)

    conv2_1 = conv4(w[5], pool1; padding=1, mode=mode) .+ w[6]
    conv2_1 = relu(conv2_1)
    conv2_2 = conv4(w[7], conv2_1; padding=1, mode=mode) .+ w[8]
    conv2_2 = relu(conv2_2)
    pool2   = pool(conv2_2)

    conv3_1 = conv4(w[9], pool2; padding=1, mode=mode) .+ w[10]
    conv3_1 = relu(conv3_1)
    conv3_2 = conv4(w[11], conv3_1; padding=1, mode=mode) .+ w[12]
    conv3_2 = relu(conv3_2)
    conv3_3 = conv4(w[13], conv3_2; padding=1, mode=mode) .+ w[14]
    conv3_3 = relu(conv3_3)
    conv3_4 = conv4(w[15], conv3_3; padding=1, mode=mode) .+ w[16]
    conv3_4 = relu(conv3_4)
    pool3   = pool(conv3_4)

    conv4_1 = conv4(w[17], pool3; padding=1, mode=mode) .+ w[18]
    conv4_1 = relu(conv4_1)
    conv4_2 = conv4(w[19], conv4_1; padding=1, mode=mode) .+ w[20]
    conv4_2 = relu(conv4_2)
    conv4_3 = conv4(w[21], conv4_2; padding=1, mode=mode) .+ w[22]
    conv4_3 = relu(conv4_3)
    conv4_4 = conv4(w[23], conv4_3; padding=1, mode=mode) .+ w[24]
    conv4_4 = relu(conv4_4)
    pool4   = pool(conv4_4)

    conv5_1 = conv4(w[25], pool4; padding=1, mode=mode) .+ w[26]
    conv5_1 = relu(conv5_1)
    conv5_2 = conv4(w[27], conv5_1; padding=1, mode=mode) .+ w[28]
    conv5_2 = relu(conv5_2)
    conv5_3 = conv4(w[29], conv5_2; padding=1, mode=mode) .+ w[30]
    conv5_3 = relu(conv5_3)
    conv5_4 = conv4(w[31], conv5_3; padding=1, mode=mode) .+ w[32]
    conv5_4 = relu(conv5_4)

    if !featuremaps
        pool5 = pool(conv5_4)
        fc6 = w[33] * mat(pool5) .+ w[34]
        fc6 = relu(fc6)
        fc6 = dropout(fc6, fc6drop)
        fc7 = w[35] * mat(fc6) .+ w[36]
        fc7 = relu(fc7)
    end
end

function get_vgg_weights(vggmat; last_layer="relu7")
    ws = []
    for l in vggmat["layers"]
        if haskey(l, "weights") && length(l["weights"]) != 0
            w = l["weights"]
            if startswith(l["name"], "conv")
                w[2] = reshape(w[2], (1,1,length(w[2]),1))
            elseif startswith(l["name"], "fc")
                w[1] = mat(w[1])'
            end
            push!(ws, w...)
        end

        last_layer == l["name"] && break
    end
    map(KnetArray, ws)
end

# resnet models
# mode, 0=>train, 1=>test
function resnet50(w,x,ms; mode=1)
    # layer 1
    conv1  = conv4(w[1],x; padding=3, stride=2) .+ w[2]
    bn1    = batchnorm(w[3:4],conv1,ms; mode=mode)
    pool1  = pool(bn1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[5:34], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[35:73], r2, ms; mode=mode)
    r4 = reslayerx5(w[74:130], r3, ms; mode=mode) # 5
    r5 = reslayerx5(w[131:160], r4, ms; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=7, mode=2)
    fc1000 = w[161] * mat(pool5) .+ w[162]
end

# mode, 0=>train, 1=>test
function resnet101(w,x,ms; mode=1)
    # layer 1
    conv1 = reslayerx1(w[1:3],x,ms; padding=3, stride=2, mode=mode)
    pool1 = pool(conv1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[4:33], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[34:72], r2, ms; mode=mode)
    r4 = reslayerx5(w[73:282], r3, ms; mode=mode)
    r5 = reslayerx5(w[283:312], r4, ms; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=7, mode=2)
    fc1000 = w[313] * mat(pool5) .+ w[314]
end

# mode, 0=>train, 1=>test
function resnet152(w,x,ms; mode=1)
    # layer 1
    conv1 = reslayerx1(w[1:3],x,ms; padding=3, stride=2, mode=mode)
    pool1 = pool(conv1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[4:33], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[34:108], r2, ms; mode=mode)
    r4 = reslayerx5(w[109:435], r3, ms; mode=mode)
    r5 = reslayerx5(w[436:465], r4, ms; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=7, mode=2)
    fc1000 = w[466] * mat(pool5) .+ w[467]
end

# mode, 0=>train, 1=>test
function resnet50(w,x,ms; mode=1)
    # layer 1
    conv1  = conv4(w[1],x; padding=3, stride=2) .+ w[2]
    bn1    = batchnorm(w[3:4],conv1,ms; mode=mode)
    pool1  = pool(bn1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[5:34], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[35:73], r2, ms; mode=mode)
    r4 = reslayerx5(w[74:130], r3, ms; mode=mode) # 5
    r5 = reslayerx5(w[131:160], r4, ms; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=7, mode=2)
    fc1000 = w[161] * mat(pool5) .+ w[162]
end

# mode, 0=>train, 1=>test
function resnet101(w,x,ms; mode=1)
    # layer 1
    conv1 = reslayerx1(w[1:3],x,ms; padding=3, stride=2, mode=mode)
    pool1 = pool(conv1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[4:33], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[34:72], r2, ms; mode=mode)
    r4 = reslayerx5(w[73:282], r3, ms; mode=mode)
    r5 = reslayerx5(w[283:312], r4, ms; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=7, mode=2)
    fc1000 = w[313] * mat(pool5) .+ w[314]
end

# mode, 0=>train, 1=>test
function resnet152(w,x,ms; mode=1)
    # layer 1
    conv1 = reslayerx1(w[1:3],x,ms; padding=3, stride=2, mode=mode)
    pool1 = pool(conv1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[4:33], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[34:108], r2, ms; mode=mode)
    r4 = reslayerx5(w[109:435], r3, ms; mode=mode)
    r5 = reslayerx5(w[436:465], r4, ms; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=7, mode=2)
    fc1000 = w[466] * mat(pool5) .+ w[467]
end

# Batch Normalization Layer
# works both for convolutional and fully connected layers
# mode, 0=>train, 1=>test
function batchnorm(w, x, ms; mode=1, epsilon=1e-5)
    mu, sigma = nothing, nothing
    if mode == 0
        d = ndims(x) == 4 ? (1,2,4) : (2,)
        s = reduce((a,b)->a*size(x,b), d)
        mu = sum(x,d) / s
        sigma = sqrt(epsilon + (sum(x.-mu,d).^2) / s)
    elseif mode == 1
        mu = shift!(ms)
        sigma = shift!(ms)
    end

    # we need getval in backpropagation
    push!(ms, AutoGrad.getval(mu), AutoGrad.getval(sigma))
    xhat = (x.-mu) ./ sigma
    return w[1] .* xhat .+ w[2]
end

function reslayerx0(w,x,ms; padding=0, stride=1, mode=1)
    b  = conv4(w[1],x; padding=padding, stride=stride)
    bx = batchnorm(w[2:3],b,ms; mode=mode)
end

function reslayerx1(w,x,ms; padding=0, stride=1, mode=1)
    relu(reslayerx0(w,x,ms; padding=padding, stride=stride, mode=mode))
end

function reslayerx2(w,x,ms; pads=[0,1,0], strides=[1,1,1], mode=1)
    ba = reslayerx1(w[1:3],x,ms; padding=pads[1], stride=strides[1], mode=mode)
    bb = reslayerx1(w[4:6],ba,ms; padding=pads[2], stride=strides[2], mode=mode)
    bc = reslayerx0(w[7:9],bb,ms; padding=pads[3], stride=strides[3], mode=mode)
end

function reslayerx3(w,x,ms; pads=[0,0,1,0], strides=[2,2,1,1], mode=1) # 12
    a = reslayerx0(w[1:3],x,ms; stride=strides[1], padding=pads[1], mode=mode)
    b = reslayerx2(w[4:12],x,ms; strides=strides[2:4], pads=pads[2:4], mode=mode)
    relu(a .+ b)
end

function reslayerx4(w,x,ms; pads=[0,1,0], strides=[1,1,1], mode=1)
    relu(x .+ reslayerx2(w,x,ms; pads=pads, strides=strides, mode=mode))
end

function reslayerx5(w,x,ms; strides=[2,2,1,1], mode=1)
    x = reslayerx3(w[1:12],x,ms; strides=strides, mode=mode)
    for k = 13:9:length(w)
        x = reslayerx4(w[k:k+8],x,ms; mode=mode)
    end
    return x
end

function get_params(params)
    len = length(params["value"])
    ws, ms = [], []
    for k = 1:len
        name = params["name"][k]
        value = convert(Array{Float32}, params["value"][k])

        if endswith(name, "moments")
            push!(ms, reshape(value[:,1], (1,1,size(value,1),1)))
            push!(ms, reshape(value[:,2], (1,1,size(value,1),1)))
        elseif startswith(name, "bn")
            push!(ws, reshape(value, (1,1,length(value),1)))
        elseif startswith(name, "fc") && endswith(name, "filter")
            push!(ws, transpose(reshape(value,size(value,3,4))))
        elseif startswith(name, "conv") && endswith(name, "bias")
            push!(ws, reshape(value, (1,1,length(value),1)))
        else
            push!(ws, value)
        end
    end
    map(KnetArray, ws), map(KnetArray, ms)
end
