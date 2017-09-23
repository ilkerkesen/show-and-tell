function get_entries(file, splits)
    file = abspath(file)
    entries = JSON.parsefile(file)["images"]
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
