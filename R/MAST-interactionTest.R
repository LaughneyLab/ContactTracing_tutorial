
require("MAST")
require("SingleCellExperiment", quietly=TRUE)
require("rhdf5")
require("data.table")
require("parallel")
require("argparse")
options(mc.cores=detectCores())

## Get infile from command line
parser <- ArgumentParser()
parser$add_argument("-i", "--infile", type="character",
                    default="/workdir/ContactTracing/tutorial/test.h5ad",
                    help="input file (h5ad).  Should have fields: .layers[\"logX\"].h5ad with log-normalized counts for each cell/gene")
parser$add_argument("-o", "--outdir", type="character",
                    default="/workdir/ContactTracing/tutorial/outdir",
                    help="output directory")
parser$add_argument("-g", "--genes", type="character",
                    help="comma-separated list of genes to run interaction test on")
parser$add_argument("-C", "--numcores", type="integer", default=1, help="number of cores to use")
parser$add_argument("-f", "--force", action="store_true", default=FALSE, help="Overwrite output files (default: do not re-run MAST if output already exists)")
parser$add_argument("-c", "--conditions", nargs=2, help="two conditions to compare in interaction test")
parser$add_argument("-V", "--covariates", type="character", default="cngeneson", help="Other columns in .obs to be used as covariate (comma-delimited)")
parser$add_argument("-p", "--populationtest", action="store_true", default=FALSE, help="Perform cluster test in addition to interaction test. (This is not necessary for contactTracing and takes a long time.)")

args <- parser$parse_args()

infile <- args$infile
cat("Input file = ", infile, "\n")
outdir <- args$outdir
cat("Output directory = ", outdir, "\n")
system(sprintf("mkdir -p %s", outdir))
force <- args$force
if (is.null(args$genes)) {
    stop("Error: Need to specify at least one gene")
}
genes <- strsplit(args$genes, ",")[[1]]
cat("Num genes to test: " , length(genes), "\n")
cat("Conditions to compare: ", args$conditions[1], args$conditions[2], "\n")
cat("Number of cores to use: ", args$numcores, "\n")
options(mc.cores=args$numcores)


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

counts <- h5read(infile, "/layers/logX")
cellNames <- as.character(h5read(infile, "/obs/_index"))
geneNames <- as.character(h5read(infile, "/var/_index"))
conditions <- as.character(readObsFromh5File(infile, "condition"))

covariates <- list()
if (!is.null(args$covariates)) {
    covariateList <- strsplit(args$covariates, ",")[[1]]
    for (v in covariateList) {
        if (v != "cngeneson") {
            covariates[[v]] <- readObsFromh5File(infile, v)
        } else {
            covariates[[v]] <- scale(colSums(counts > 0))
        }
        cat("Using", v, "as a covariate\n")
    }
}


# check first that we have all genes
for (gene in genes) {
    if (length(which(geneNames == gene)) != 1) {
        stop("Error finding gene ", gene, "\n")
    }
}

sca <- FromMatrix(counts, 
                  cData=data.frame(wellKey=cellNames, barcode=cellNames, condition=conditions),
                  fData=data.frame(primerid=geneNames, geneName=geneNames))
for (v in names(covariates)) {
     colData(sca)[[v]] <- covariates[[v]]
}

writeResults <- function(outdir, gene, testname, zt, zlr=NULL) {
  cons <- unique(zt$contrast)
  if (!is.null(zlr)) {
    zlr0 <- zlr[,"hurdle","Pr(>Chisq)"]
    df <- data.frame(primerid=names(zlr0), pval=as.numeric(zlr0))
  } else {
    pcontrast <- unique(zt[!is.na(zt$`Pr(>Chisq)`),'contrast']$contrast)
    if (length(pcontrast) != 1) {
      stop("Multiple contrasts have p-values, need to pass zlr to writeResults")
    }
    f <- zt$contrast == pcontrast & zt$component=='H'
    df <- data.frame(primerid=zt[f,]$primerid, pval=zt[f,]$`Pr(>Chisq)`)
  }
  for (con in cons) {
    if (con == '(Intercept)') next
    currdf <- zt[contrast == con & component=='logFC', .(primerid, coef, ci.hi, ci.lo)]
    names(currdf)[2:4] <- sprintf("%s_%s", names(currdf)[2:4], con)
    df <- merge(df, currdf, by='primerid')
  }
  df <- df[order(df$pval),]
  if (!is.null(outdir)) {
    curroutdir <- sprintf("%s/%s", outdir, gene)
    system(sprintf("mkdir -p %s", curroutdir))
    write.table(df, file=sprintf("%s/%s.txt", curroutdir, testname), quote=FALSE, sep="\t", row.names=FALSE)
  }
  invisible(df)
}

testname <- sprintf("interaction_%s_vs_%s",args$conditions[1], args$conditions[2])
for (gene in genes) {
    cat("Running interaction test on", gene, "expression\n")
    results <- list()
    w <- which(geneNames == gene)
    isExpressed <- counts[w,] > 0
    gene <- gsub(" ", "_", gene)
    colData(sca)$cluster <- factor(isExpressed, levels=c(FALSE, TRUE))
    numInCluster <- sum(colData(sca)$cluster=='TRUE')
    cat(numInCluster/ncol(counts), "fraction of cells expressing", gene, "\n")
    if (numInCluster < 2 || numInCluster >= ncol(counts)) {
        cat("...skipping\n")
        next
    }
    doskip=FALSE
    for (condition in args$conditions) {
        numInCluster <- sum(colData(sca[,condition==conditions])$cluster=='TRUE')
        numOutOfCluster <- sum(condition == conditions) - numInCluster
        if (numInCluster < 1 || numOutOfCluster < 1) {
            cat(condition, numInCluster, "in cluster... ", numOutOfCluster, "not in cluster... skipping interaction test for", gene, "\n")
            doskip=TRUE
        }
    }
    if (doskip) {
        next
    }
    clusterName <- sprintf("cluster_%s", args$conditions[1])
    colData(sca)[,clusterName] <- factor(isExpressed & conditions==args$conditions[1], levels=c(FALSE, TRUE))
    colData(sca)[,args$conditions[2]] <- factor(conditions == args$conditions[2], levels=c(FALSE, TRUE))
    outfile <- sprintf("%s/%s/%s.txt", outdir, gene, testname)
    if (file.exists(outfile) && !force) {
        cat("Skipping gene", gene, "... outfile", outfile, "already exists\n")
        next
    }
    group1 <- (conditions == args$conditions[1]) & isExpressed
    group2 <- (conditions == args$conditions[1]) & (!isExpressed)
    group3 <- (conditions == args$conditions[2]) & isExpressed
    group4 <- (conditions == args$conditions[2]) & (!isExpressed)
              
    count1 <- apply(counts[,group1,drop=FALSE] > 0, 1, sum)
    count2 <- apply(counts[,group2,drop=FALSE] > 0, 1, sum)
    count3 <- apply(counts[,group3,drop=FALSE] > 0, 1, sum)
    count4 <- apply(counts[,group4,drop=FALSE] > 0, 1, sum)

    keepGenes <- rep(TRUE, length(count1))
    keepGenes[count1==0 & count2==0] <- FALSE
    keepGenes[count1==0 & count3==0] <- FALSE
    keepGenes[count2==0 & count4==0] <- FALSE
    keepGenes[count3==0 & count4==0] <- FALSE
    
    cat("keeping" , sum(keepGenes) ,  "out of" ,  length(keepGenes) , "genes\n")
    if (sum(keepGenes) == 0) {
        next
    }
              
    cat("counts:", sum(isExpressed), sum(!isExpressed), sum(conditions==args$conditions[1]), sum(conditions==args$conditions[2]), "\n")
    print(table(isExpressed[is.element(conditions, args$conditions)], conditions[is.element(conditions, args$conditions)]))
    if (sum(isExpressed & conditions == args$conditions[2]) == 0) {
        cat("No expressed/condition2... skipping\n")
        next
    }
    t1 <- Sys.time()
    print(t1)
    zlmstr <- sprintf("~%s + cluster + %s", args$conditions[2], clusterName)
    for (v in names(covariates)) {
        zlmstr <- sprintf("%s + %s", zlmstr, v)
    }
    zlmResults <- zlm(formula(zlmstr), 
                      sca[keepGenes,is.element(sca$condition, args$conditions)])
    t2 <- Sys.time()
    cat("t2 - t1\n")
    print(t2-t1)
    zt <- summary(zlmResults)$datatable
    t3 <- Sys.time()
    print(t3)
    cat("t3 - t2\n")
    print(t3 - t2)
    zlr <- lrTest(zlmResults, clusterName)
    t4 <- Sys.time()
    print(t4)
    cat("t4 - t3\n")
    print(t4 - t3)
    results[[testname]] <- writeResults(outdir, gene, testname, zt, zlr)
}

if (args$populationtest) {  # original cluster test code, will use faster test in scanpy instead
    for (gene in genes) {
        cat("Running population test on", gene, "expression\n")
        results <- list()
        w <- which(geneNames == gene)
        isExpressed <- counts[w,] > 0
        gene <- gsub(" ", "_", gene)
        colData(sca)$cluster <- factor(isExpressed, levels=c(FALSE, TRUE))
        numInCluster <- sum(colData(sca)$cluster=='TRUE')
        cat(numInCluster/ncol(counts), "fraction of cells expressing", gene, "\n")
        if (numInCluster < 2 || numInCluster >= ncol(counts)) {
            cat("...skipping\n")
            next
        }
        testname <- "population_test"
        if (force || !file.exists(sprintf("%s/%s/%s.txt", outdir, gene, testname))) {
            # test for whether there is any cluster effect
            cat(testname, "\n")
            t1 <- Sys.time()
            print(t1)
            zlmstr <-"~cluster"
            for (v in names(covariates)) {
                zlmstr <- sprintf("%s + %s", zlmstr, v)
            }
            zlmResults <- zlm(formula(zlmstr), sca)
            t2 <- Sys.time()
            print(t2)
            cat("t2 - t1\n")
            print(t2-t1)
            zt <- summary(zlmResults)$datatable
            t3 <- Sys.time()
            print(t3)
            cat("t3 - t2\n")
            print(t3 - t2)
            zlr <- lrTest(zlmResults, "cluster")
            t4 <- Sys.time()
            print(t4)
            cat("t4 - t3\n")
            print(t4 - t3)
            results[[testname]] <- writeResults(outdir, gene, testname, zt, zlr)
        }
    }
}


