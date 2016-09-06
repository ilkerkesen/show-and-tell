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
