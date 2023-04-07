require("MAST")
require("SingleCellExperiment", quietly=TRUE)
require("rhdf5")
require("data.table")
require("argparse")
require("parallel")
options(mc.cores=detectCores())

## Get infile from command line
parser <- ArgumentParser()
parser$add_argument("-i", "--infile", type="character",
                    default="/workdir/ContactTracing/tutorial/test.h5ad",
                    help="input file (h5ad).  Should have fields: .layers[\"logX\"].h5ad with log-normalized counts for each cell/gene")
parser$add_argument("-o", "--outdir", type="character",
                    default="/workdir/ContactTracing/tutorial/outdir",
                    help="output directory")
parser$add_argument("-g", "--groups", type="character",
                    default="condition",
                    help="column in .obs that will be used to group cells")
parser$add_argument("-c", "--comp-groups", nargs=2)
parser$add_argument("-C", "--numcore", type="integer", default=1, help="number of cores to use")
parser$add_argument("-f", "--force", action="store_true", default=FALSE, help="Overwrite output files (default: do not re-run MAST if output already exists)")
parser$add_argument("-V", "--covariates", type="character", default="cngeneson", help="Other columns in .obs to be used as covariate (comma-delimited)")

args <- parser$parse_args()

infile <- args$infile
cat("Input file = ", infile, "\n")
outdir <- args$outdir
cat("Output directory = ", outdir, "\n")
system(sprintf("mkdir -p %s", outdir))

groupName <- args$groups
cat("grouping by ", groupName, "\n")

compGroups <- args$comp_groups
if (is.null(compGroups)) {
    cat("Comparing all values of", groupName, "to each other\n")
} else {
    cat("Comparing", compGroups[1], "to", compGroups[2], "\n")
}


cat("Using ", args$numcore, "cores\n")
options(mc.cores=args$numcore)

counts <- h5read(infile, "/layers/logX")
if (is.null(counts)) {
    stop("Error: no data in /layers/logX")
}


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


cellNames <- as.character(h5read(infile, "/obs/_index"))
geneNames <- as.character(h5read(infile, "/var/_index"))

clusters <- readObsFromh5File(infile, groupName)
names(clusters) <- cellNames


if (length(compGroups) == 2) {
    if (compGroups[2] == 'others') {
        keep <- rep(TRUE, length(cellNames))
    } else {
        keep <- (clusters == compGroups[1] | clusters == compGroups[2])
        clusters <- clusters[keep]
        counts <- counts[,keep]
        cellNames <- cellNames[keep]
    }
    clusterList <- compGroups[1]
} else {
    clusterList <- unique(as.character(clusters))
    keep <- rep(TRUE, length(cellNames))
}


covariates <- list()
if (!is.null(args$covariates)) {
    covariateList <- strsplit(args$covariates, ",")[[1]]
    for (v in covariateList) {
        if (v != "cngeneson") {
            covariates[[v]] <- readObsFromh5File(infile, v)[keep]
        } else {
            covariates[[v]] <- scale(colSums(counts > 0))
        }
        cat("Using", v, "as a covariate\n")
    }
}


fixNAs <- function(fcHurdle, cluster, clusters, geneNames, counts) {
    pcol <- grep("Chisq", names(fcHurdle), value=TRUE)
    if (length(pcol) != 1) {
        cat("Error finding p-value column in fixNAs")
        return(fcHurdle)
    }
    fixNA <- (is.na(fcHurdle$coef) & fcHurdle[,pcol] < 1)
    coefRange <- range(fcHurdle$coef, na.rm=TRUE)
    geneOrder <- sapply(fcHurdle$primerid, function(x) {which(geneNames ==x)})
    fracIn <- apply(counts[geneOrder,clusters==cluster], 1, sum)/sum(clusters==cluster)
    fracOut <- apply(counts[geneOrder,clusters != cluster], 1, sum)/sum(clusters!=cluster)
    fcHurdle[fixNA & fracIn <= fracOut,"coef"] <- coefRange[1]
    fcHurdle[fixNA & fracIn > fracOut, "coef"] <- coefRange[2]
    fcHurdle
}





for (i in 1:length(clusterList)) {
    if (class(clusterList) == "list") {
        clusters <- clusterList[[i]]
        cluster <- names(clusterList)[i]
    } else {
        cluster <- clusterList[i]
    }
    clusterName <- gsub("/","_", cluster, fixed=TRUE)
    clusterName2 <- sprintf("cluster%s", cluster)
    clusterAssign <- ifelse(clusters == cluster, cluster, "background")
                                        # get outfile name
    if (length(compGroups) == 2) {
        outfile <- sprintf("%s/%s_vs_%s.csv", outdir, compGroups[1], compGroups[2])
    } else {
        outfile <- sprintf("%s/%s%.csv", outdir, clusterName)
    }
    outfile <- gsub(" ", "_", outfile, fixed=TRUE)
    cat("outfile = \"", outfile, "\"\n", sep="")
    
    if ((!args$force) && file.exists(outfile)) {
        cat("outfile already exists, skipping\n")
        next
    }
    
    cat("Running MAST on ", clusterName, "\n")
    sca <- FromMatrix(counts, 
                      cData=data.frame(wellKey=cellNames, barcode=cellNames, cluster=clusterAssign), 
                      fData=data.frame(primerid=geneNames, geneName=geneNames))
    tmpc <- factor(colData(sca)$cluster)
    tmpc <- relevel(tmpc, "background")
    colData(sca)$cluster <- tmpc
                                        #  colData(sca)$cngeneson <- scale(colSums(assay(sca) > 0))
    for (v in names(covariates)) {
        colData(sca)[[v]] <- covariates[[v]]
    }


    zlmstr <- ("~cluster")
    for (v in names(covariates)) {
        zlmstr <- sprintf("%s + %s", zlmstr, v)
    }
    zlmResults <- zlm(formula(zlmstr), sca)
    zt <- summary(zlmResults, doLRT=clusterName2)$datatable
    
    fcHurdle <- merge(zt[contrast==clusterName2 & component=='H',.(primerid, `Pr(>Chisq)`)],
                      zt[contrast==clusterName2 & component=='logFC', .(primerid, coef, ci.hi, ci.lo)], by='primerid')
    
    fcHurdle[,fdr:=p.adjust(`Pr(>Chisq)`, 'fdr')]
    setorder(fcHurdle, fdr)
    fcHurdle <- fixNAs(as.data.frame(fcHurdle), cluster, clusters, geneNames, counts)
    
    write.csv(fcHurdle, outfile, quote=FALSE)
}



