function get_entries(zip, splits)
    zip = abspath(zip)
    file = joinpath(splitext(splitdir(abspath(zip))[2])[1], "dataset.json")
    entries = JSON.parse(readstring(`unzip -p $zip $file`))["images"]
    return map(s->filter(x->x["split"]==s, entries), splits)
end

function get_pairs(entries)
    pairs = []
    for entry in entries
        for sentence in entry["sentences"]
            push!(pairs, (entry["filename"], sentence["tokens"]))
        end
    end
    return pairs
end
