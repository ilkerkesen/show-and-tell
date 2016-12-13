# VGG16 model for convolutional feature etraction
function vgg16(w, x; pdrop=0.0, mode=0, featuremaps=false)
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
    pool5   = pool(conv5_3)

    fc6 = w[27] * mat(pool5) .+ w[28]
    fc6 = relu(fc6)
    if pdrop > 0
        fc6 = dropout(fc6, pdrop)
    end

    fc7 = w[29] * mat(fc6) .+ w[30]
    fc7 = relu(fc7)
    if pdrop > 0
        fc7 = dropout(fc7, pdrop)
    end

    if !featuremaps
        return fc7
    else
        return fc7, pool5
    end
end


function get_vgg_weights(vggmat; last_layer=nothing)
    ws = []
    for l in vggmat["layers"]
        if haskey(l, "weights") && length(l["weights"]) != 0
            w = l["weights"]
            if startswith(l["name"], "conv")
                w[2] = reshape(w[2], (1,1,length(w[2]),1))
            elseif startswith(l["name"], "fc")
                w[1] = transpose(mat(w[1]))
            end
            push!(ws, w...)
        end

        last_layer == l["name"] && break
    end
    map(KnetArray, ws)
end
