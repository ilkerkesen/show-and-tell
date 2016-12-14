function extract_file_from_tar(tar, from, to)
    sc = count(x->x=='/', from[1:end-1])
    run(`tar -xf $tar --strip-components=$sc -C $to --directory $from`)
end

function extract_file_from_zip(zip, from, to)
    filecontent = readstring(`unzip -p $zip $from`)
    write(to, filecontent)
end

# FIXME: pretty elapsed time conversion
function pretty_time(t)
    t = round(t)
    h = Int(div(t, 3600))
    m = Int(div(mod(t, 3600), 60))
    s = Int(mod(mod(t, 3600), 60))

    retval = ""
    h != 0 && (retval = string(retval, h, " hours "))
    m != 0 && (retval = string(retval, m, " minutes "))
    s != 0 && (retval = string(retval, s, " seconds"))

    return retval
end
