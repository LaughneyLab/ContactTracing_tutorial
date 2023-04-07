
## given h5ad anndata file, return given column from adata.obs
readObsFromh5File <- function(infile, colname) {
    tmp <- h5ls(infile)
    w <- which(tmp$name == colname & tmp$otype=="H5I_DATASET")
    if (length(w) == 1) {
        return(h5read(infile, sprintf("/obs/%s", colname)))
    }
    w <- which(tmp$group == sprintf("/obs/%s", colname))
    if (length(w) == 1) {
        return(h5read(infile, sprintf("/obs/%s", colname)))
    }
    if (length(w) == 2) {
        w2 <- which(tmp[w,]$name=="categories")
        catNames <- h5read(infile, sprintf("%s/%s", tmp[w,][w2,"group"], tmp[w,][w2,"name"]))
        w2 <- which(tmp[w,]$name == "codes")
        return(catNames[as.numeric(h5read(infile, sprintf("%s/%s", tmp[w,][w2, "group"], tmp[w,][w2, "name"])))+1])
    }
    stop("Error reading ", colname, " from ", infile)
}

