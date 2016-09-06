# FIXME: pretty elapsed time conversion
function pretty_time(t)
    t = round(t)
    h = div(t, 3600)
    m = div(mod(t, 3600), 60)
    s = mod(mod(t, 3600), 60)

    retval = ""
    h != 0 && (retval = string(retval, h, " hours "))
    m != 0 && (retval = string(retval, m, " minutes "))
    s != 0 && (retval = string(retval, s, " seconds"))

    return retval
end
