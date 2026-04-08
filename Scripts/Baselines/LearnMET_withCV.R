################################################################################
# learnMET evaluation that matches your Python baselines:
#   Tier-1: genotype-disjoint GroupKFold CV (no genotype leakage)
#   Tier-2: within-genotype environment holdout CV
# Metrics: R2 (sklearn style), RMSE, MAE, CCC + pooled (micro) metrics
################################################################################
setwd("/Users/thulanihewavithana/Documents/PhD/Research/EcoPopDL_GP/")
library(dplyr)
library(learnMET)

# ----------------------------- configuration ----------------------------------
N_FOLDS <- 4
RANDOM_STATE <- 20

TIER2_TEST_FRAC <- 0.2
TIER2_VAL_FRAC  <- 0.2
TIER2_SEED_OFFSET <- 1000

LEARNMET_METHOD <- "xgb_reg_1"               # your learnMET model
CV_OUT_ROOT <- "learnMET_output/cv_learnMET" # outputs per fold go here

# Match Python-style target standardization inside each fold (optional).
# Your Python baselines standardize y in train and then de-standardize preds.
STANDARDIZE_Y <- TRUE

# If you want to match your Python feature policy (NO Year/Location dummy leakage),
# set these FALSE. If you want your current learnMET settings, flip them TRUE.
YEAR_INCLUDED <- FALSE
LOCATION_INCLUDED <- FALSE
LATLON_INCLUDED <- FALSE

INCLUDE_ENV_PREDICTORS <- TRUE    # your env_cov numeric columns (MaxTemp/MinTemp/Precip/etc)
INCLUDE_GXE <- TRUE
GXE_INCLUDED <- TRUE

NUM_PCS <- NULL                   # keep as in your current learnMET call
TYPE_LOCATION_INFO <- "categorical"
# -----------------------------------------------------------------------------


# ---- metrics ----------------------------------------------------------------
var_pop <- function(x) {
  x <- as.numeric(x)
  m <- mean(x)
  mean((x - m)^2)
}

ccc <- function(y_true, y_pred) {
  y_true <- as.numeric(y_true)
  y_pred <- as.numeric(y_pred)
  ok <- is.finite(y_true) & is.finite(y_pred)
  y_true <- y_true[ok]; y_pred <- y_pred[ok]
  if (length(y_true) == 0) return(NA_real_)
  
  mean_true <- mean(y_true); mean_pred <- mean(y_pred)
  var_true <- var_pop(y_true)                   # pop var (matches np.var default)
  var_pred <- var_pop(y_pred)
  cov_tp <- cov(y_true, y_pred)                 # sample cov (matches np.cov default)
  denom <- var_true + var_pred + (mean_true - mean_pred)^2
  if (!is.finite(denom) || denom <= 0) return(NA_real_)
  as.numeric(2 * cov_tp / denom)
}

metrics <- function(y_true, y_pred) {
  y_true <- as.numeric(y_true)
  y_pred <- as.numeric(y_pred)
  ok <- is.finite(y_true) & is.finite(y_pred)
  y_true <- y_true[ok]; y_pred <- y_pred[ok]
  if (length(y_true) == 0) {
    return(list(r2=NA_real_, rmse=NA_real_, mae=NA_real_, ccc=NA_real_))
  }
  
  # sklearn-style R2 = 1 - SSE/SST
  sst <- sum((y_true - mean(y_true))^2)
  sse <- sum((y_true - y_pred)^2)
  r2  <- if (is.finite(sst) && sst > 0) 1 - sse / sst else NA_real_
  
  rmse <- sqrt(mean((y_true - y_pred)^2))
  mae  <- mean(abs(y_true - y_pred))
  list(r2=r2, rmse=rmse, mae=mae, ccc=ccc(y_true, y_pred))
}

standardize_target <- function(y) {
  y <- as.numeric(y)
  m <- mean(y, na.rm=TRUE)
  s <- sqrt(mean((y - m)^2, na.rm=TRUE))   # pop SD (matches np.std default)
  if (!is.finite(s) || s < 1e-8) s <- 1
  list(z=(y - m)/s, mean=m, sd=s)
}


# ---- folds -------------------------------------------------------------------
group_kfold_indices <- function(groups, k) {
  # Replicates sklearn GroupKFold assignment logic (greedy balance by group counts)
  groups <- as.character(groups)
  uniq <- sort(unique(groups))             # matches np.unique sorting
  g_idx <- match(groups, uniq)             # 1..n_groups
  n_groups <- length(uniq)
  counts <- tabulate(g_idx, nbins=n_groups)
  
  # sort by counts desc; tie-break by group index desc (like argsort()[::-1])
  sorted_g <- order(counts, seq_along(counts), decreasing=TRUE)
  
  fold_counts <- rep(0L, k)
  fold_groups <- vector("list", k)
  for (gi in sorted_g) {
    fi <- which.min(fold_counts)
    fold_groups[[fi]] <- c(fold_groups[[fi]], gi)
    fold_counts[fi] <- fold_counts[fi] + counts[gi]
  }
  
  folds <- vector("list", k)
  for (fi in seq_len(k)) {
    test_idx <- which(g_idx %in% fold_groups[[fi]])
    train_idx <- setdiff(seq_along(groups), test_idx)
    folds[[fi]] <- list(train=train_idx, test=test_idx)
  }
  folds
}

within_genotype_env_folds <- function(genotypes, n_folds, test_frac, val_frac, seed) {
  genotypes <- as.character(genotypes)
  unique_genos <- sort(unique(genotypes))
  folds <- vector("list", n_folds)
  
  for (fold in seq_len(n_folds)) {
    set.seed(seed + (fold-1) * 101)
    tr_idx <- integer(); va_idx <- integer(); te_idx <- integer()
    
    for (gid in unique_genos) {
      idxs <- which(genotypes == gid)
      if (length(idxs) == 0) next
      perm <- sample(idxs, length(idxs), replace=FALSE)
      n <- length(perm)
      
      if (n < 2) {
        tr_idx <- c(tr_idx, perm)
        next
      }
      
      n_test <- max(1L, ceiling(test_frac * n))
      n_val  <- max(0L, ceiling(val_frac  * n))
      
      if (n_test + n_val >= n) {
        n_val <- min(n_val, n - n_test - 1L)
        if (n_val < 0) n_val <- 0L
        if (n_test + n_val >= n) {
          n_test <- max(1L, n - n_val - 1L)
        }
      }
      
      test_i <- perm[seq_len(n_test)]
      val_i  <- if (n_val > 0) perm[(n_test+1):(n_test+n_val)] else integer()
      train_i <- perm[(n_test+n_val+1):n]
      
      if (length(train_i) == 0) {
        train_i <- test_i[1]
        test_i <- test_i[-1]
      }
      
      tr_idx <- c(tr_idx, train_i)
      va_idx <- c(va_idx, val_i)
      te_idx <- c(te_idx, test_i)
    }
    
    folds[[fold]] <- list(train=sort(tr_idx), val=sort(va_idx), test=sort(te_idx))
  }
  folds
}


# ---- environment prep per fold ------------------------------------------------
prepare_env_inputs <- function(pheno_df, env_cov_env, loc_coords, trait) {
  
  pheno_df <- pheno_df %>%
    mutate(
      geno_ID  = trimws(as.character(geno_ID)),
      location = trimws(as.character(location)),
      year     = as.numeric(year)
    ) %>%
    select(geno_ID, year, location, all_of(trait))
  
  pheno_df[[trait]] <- as.numeric(pheno_df[[trait]])
  
  envs <- unique(pheno_df[, c("year", "location")])
  
  loc2 <- loc_coords
  loc2$location <- trimws(as.character(loc2$location))
  loc2 <- loc2[!duplicated(loc2$location), , drop = FALSE]
  
  info_env <- merge(envs, loc2, by = "location", all.x = TRUE)
  info_env <- info_env[, c("year", "location", "longitude", "latitude")]
  info_env$year <- as.numeric(info_env$year)
  info_env$location <- as.character(info_env$location)
  
  climate <- merge(envs, env_cov_env, by = c("year", "location"), all.x = TRUE)
  
  list(
    pheno = as.data.frame(pheno_df),
    info_env = as.data.frame(info_env),
    climate = as.data.frame(climate)
  )
}





# ---- run one fold -------------------------------------------------------------
run_learnmet_fold <- function(
    pheno_all,
    geno_mat,
    env_cov_env,
    loc_coords,
    trait,
    train_idx,
    test_idx,
    fold_tag,
    out_root,
    prediction_method = "xgb_reg_1",
    seed = 20,
    standardize_y = TRUE,
    YEAR_INCLUDED = FALSE,
    LOCATION_INCLUDED = FALSE,
    LATLON_INCLUDED = FALSE,
    INCLUDE_ENV_PREDICTORS = TRUE
) {
  
  # --- split
  ph_tr_raw <- pheno_all[train_idx, , drop = FALSE]
  ph_te_raw <- pheno_all[test_idx,  , drop = FALSE]
  
  # Keep only needed columns from your bigger pheno table
  # (these can exist alongside other columns in pheno_all; we just don't pass them to learnMET)
  ph_tr_raw <- ph_tr_raw[, c("geno_ID", "year", "location", trait), drop = FALSE]
  ph_te_raw <- ph_te_raw[, c("geno_ID", "year", "location", trait), drop = FALSE]
  
  # basic cleaning
  ph_tr_raw$geno_ID  <- trimws(as.character(ph_tr_raw$geno_ID))
  ph_tr_raw$location <- trimws(as.character(ph_tr_raw$location))
  ph_tr_raw$year     <- as.numeric(ph_tr_raw$year)
  ph_tr_raw[[trait]] <- as.numeric(ph_tr_raw[[trait]])
  
  ph_te_raw$geno_ID  <- trimws(as.character(ph_te_raw$geno_ID))
  ph_te_raw$location <- trimws(as.character(ph_te_raw$location))
  ph_te_raw$year     <- as.numeric(ph_te_raw$year)
  ph_te_raw[[trait]] <- as.numeric(ph_te_raw[[trait]])
  
  # drop missing trait rows (important)
  ph_tr_raw <- ph_tr_raw[!is.na(ph_tr_raw[[trait]]), , drop = FALSE]
  ph_te_true <- ph_te_raw[!is.na(ph_te_raw[[trait]]), , drop = FALSE]
  
  # filter to genotypes that exist in geno_mat rownames (learnMET requirement)
  # ph_tr_raw <- ph_tr_raw[ph_tr_raw$geno_ID %in% rownames(geno_mat), , drop = FALSE]
  ph_te_true <- ph_te_true[ph_te_true$geno_ID %in% rownames(geno_mat), , drop = FALSE]
  
  if (nrow(ph_tr_raw) == 0 || nrow(ph_te_true) == 0) {
    stop("[", fold_tag, "] Empty train/test aDTFer filtering. Check geno_ID overlap with rownames(geno_mat).")
  }
  
  # --- optional: standardize y like your Python baselines
  y_mean <- 0; y_sd <- 1
  if (standardize_y) {
    y <- ph_tr_raw[[trait]]
    y_mean <- mean(y, na.rm = TRUE)
    y_sd <- sd(y, na.rm = TRUE)
    if (!is.finite(y_sd) || y_sd < 1e-8) y_sd <- 1
    ph_tr_raw[[trait]] <- (y - y_mean) / y_sd
  }
  
  
  # --- build masked test pheno for learnMET
  ph_te_mask <- ph_te_true
  # ph_te_mask[[trait]] <- NA_real_
  
  cat("[DEBUG] overlap geno IDs:",
      sum(ph_tr_raw$geno_ID %in% rownames(geno_mat)),
      "of", nrow(ph_tr_raw), "\n")
  
  # --- fold inputs (learnMET compliant)
  env_tr <- prepare_env_inputs(ph_tr_raw, env_cov_env, loc_coords, trait)
  env_te <- prepare_env_inputs(ph_te_mask, env_cov_env, loc_coords, trait)
  
  fold_root <- file.path(out_root, fold_tag)
  dir_train <- file.path(fold_root, "train")
  dir_test  <- file.path(fold_root, "test")
  dir.create(dir_train, recursive = TRUE, showWarnings = FALSE)
  dir.create(dir_test,  recursive = TRUE, showWarnings = FALSE)
  cat("\n[DEBUG]", fold_tag, "\n")
  cat("  train rows:", nrow(ph_tr_raw), "\n")
  cat("  train non-NA trait:", sum(!is.na(ph_tr_raw[[trait]])), "\n")
  cat("  trait summary:\n")
  print(summary(ph_tr_raw[[trait]]))
  
  if (sum(!is.na(ph_tr_raw[[trait]])) == 0) {
    stop("[DEBUG] Fold has 0 non-NA training values for trait: ", trait)
  }
  
  MET_tr <- create_METData(
    geno = geno_mat,
    pheno = env_tr$pheno,
    climate_variables = env_tr$climate,
    compute_climatic_ECs = FALSE,
    info_environments = env_tr$info_env,
    map = NULL,
    path_to_save = dir_train
  )
  
  MET_te <- create_METData(
    geno = geno_mat,
    pheno = env_te$pheno,
    climate_variables = env_te$climate,
    compute_climatic_ECs = FALSE,
    info_environments = env_te$info_env,
    map = NULL,
    path_to_save = dir_test,
    as_test_set = TRUE
  )
  
  set.seed(seed)
  res <- predict_trait_MET(
    METData_training = MET_tr,
    METData_new = MET_te,
    trait = trait,
    prediction_method = prediction_method,
    include_gxe = TRUE,
    gxe_included = TRUE,
    lat_lon_included = LATLON_INCLUDED,
    year_included = YEAR_INCLUDED,
    location_included = LOCATION_INCLUDED,
    include_env_predictors = INCLUDE_ENV_PREDICTORS,
    seed = seed,
    save_model = FALSE,
    type_location_info = "categorical",
    path_folder = dir_test
  )
  
  pred_df <- res$list_results[[1]]$predictions_df
  
  cat("[DEBUG] pred_df columns:\n")
  print(names(pred_df))
  
  pred_df$geno_ID <- trimws(as.character(pred_df$geno_ID))
  
  if ("location" %in% names(pred_df)) pred_df$location <- trimws(as.character(pred_df$location))
  if ("year" %in% names(pred_df)) pred_df$year <- as.numeric(pred_df$year)
  
  # If IDenv doesn't exist, create it ONLY if year+location exist
  if (!"IDenv" %in% names(pred_df)) {
    stopifnot(all(c("location","year") %in% names(pred_df)))
    pred_df$IDenv <- paste(pred_df$location, pred_df$year, sep = "_")
  } else {
    pred_df$IDenv <- trimws(as.character(pred_df$IDenv))
  }
  
  
  # Make sure year/location are consistent types
  
  
  # Inspect what columns exist
  # print(names(pred_df))
  
  # --- clean prediction df ---
  pred_df$geno_ID <- trimws(as.character(pred_df$geno_ID))
  pred_df$IDenv   <- trimws(as.character(pred_df$IDenv))
  
  # --- build truth keys (match learnMET's IDenv exactly) ---
  ph_te_true$geno_ID  <- trimws(as.character(ph_te_true$geno_ID))
  ph_te_true$location <- trimws(as.character(ph_te_true$location))
  ph_te_true$year     <- as.numeric(ph_te_true$year)
  ph_te_true$IDenv    <- paste(ph_te_true$location, ph_te_true$year, sep = "_")
  ph_te_true$IDenv    <- trimws(as.character(ph_te_true$IDenv))
  
  # --- keep only what we need ---
  pred_key <- pred_df[, c("geno_ID", "IDenv", ".pred"), drop = FALSE]
  truth_key <- ph_te_true[, c("geno_ID", "IDenv", trait), drop = FALSE]
  
  # --- merge predictions with observed test trait ---
  eval_df <- merge(pred_key, truth_key, by = c("geno_ID", "IDenv"), all.x = TRUE)
  
  # --- extract vectors ---
  y_true <- as.numeric(eval_df[[trait]])
  y_pred <- as.numeric(eval_df$.pred)
  
  # de-standardize predictions back to original scale
  if (standardize_y) {
    y_pred <- y_pred * y_sd + y_mean
  }
  
  # --- DEBUG: do we actually have matched values? ---
  cat("[DEBUG] eval rows:", nrow(eval_df), "\n")
  cat("[DEBUG] non-NA y_true:", sum(!is.na(y_true)), "\n")
  cat("[DEBUG] non-NA y_pred:", sum(!is.na(y_pred)), "\n")
  
  # return
  list(y_true = y_true, y_pred = y_pred)

}



# ---- summarize + pooled --------------------------------------------------------
summarize_cv <- function(fold_rows) {
  fold_rows %>%
    summarize(
      cv_r2_mean   = mean(r2, na.rm=TRUE),
      cv_r2_std    = sd(r2, na.rm=TRUE),
      cv_rmse_mean = mean(rmse, na.rm=TRUE),
      cv_rmse_std  = sd(rmse, na.rm=TRUE),
      cv_mae_mean  = mean(mae, na.rm=TRUE),
      cv_mae_std   = sd(mae, na.rm=TRUE),
      cv_ccc_mean  = mean(ccc, na.rm=TRUE),
      cv_ccc_std   = sd(ccc, na.rm=TRUE)
    )
}

pooled_metrics_df <- function(y_true_all, y_pred_all) {
  m <- metrics(y_true_all, y_pred_all)
  data.frame(pooled_r2=m$r2, pooled_rmse=m$rmse, pooled_mae=m$mae, pooled_ccc=m$ccc)
}


# ---- Tier-1 CV -----------------------------------------------------------------
learnmet_tier1_cv <- function(
    pheno_all, geno_mat, env_cov, loc_coords, trait,
    n_folds = N_FOLDS, seed = RANDOM_STATE, out_root = CV_OUT_ROOT
) {
  folds <- group_kfold_indices(pheno_all$geno_ID, n_folds)
  
  fold_metrics <- data.frame()
  pooled_true <- c()
  pooled_pred <- c()
  
  for (i in seq_along(folds)) {
    cat("\n", strrep("=", 80), "\n", sep="")
    cat(sprintf("Tier-1 Fold %d/%d (genotype-disjoint CV)\n", i, n_folds))
    cat(strrep("=", 80), "\n", sep="")
    
    fold_tag <- sprintf("tier1_fold%02d", i)
    out <- run_learnmet_fold(
      pheno_all = pheno_all,
      geno_mat = geno_mat,
      env_cov = env_cov,
      loc_coords = loc_coords,
      trait = trait,
      train_idx = folds[[i]]$train,
      test_idx = folds[[i]]$test,
      fold_tag = fold_tag,
      out_root = out_root,
      seed = seed + i
    )
    
    m <- metrics(out$y_true, out$y_pred)
    
    fold_metrics <- bind_rows(
      fold_metrics,
      data.frame(fold=i, r2=m$r2, rmse=m$rmse, mae=m$mae, ccc=m$ccc)
    )
    
    pooled_true <- c(pooled_true, out$y_true)
    pooled_pred <- c(pooled_pred, out$y_pred)
    
    cat(sprintf("Fold %d metrics: R2=%.4f, RMSE=%.4f, MAE=%.4f, CCC=%.4f\n",
                i, m$r2, m$rmse, m$mae, m$ccc))
  }
  
  summary <- summarize_cv(fold_metrics)
  pooled  <- pooled_metrics_df(pooled_true, pooled_pred)
  
  dir.create(out_root, recursive=TRUE, showWarnings=FALSE)
  write.csv(fold_metrics, file.path(out_root, "learnmet_tier1_fold_metrics.csv"), row.names=FALSE)
  write.csv(summary,      file.path(out_root, "learnmet_tier1_summary.csv"), row.names=FALSE)
  write.csv(pooled,       file.path(out_root, "learnmet_tier1_pooled.csv"), row.names=FALSE)
  
  cat("\n", strrep("=", 80), "\n", sep="")
  cat("TIER-1 CV RESULTS (mean +/- sd)\n")
  cat(strrep("=", 80), "\n", sep="")
  print(summary)
  cat("\nPOOLED (micro) METRICS\n")
  print(pooled)
  
  invisible(list(fold_metrics=fold_metrics, summary=summary, pooled=pooled))
}


# ---- Tier-2 CV -----------------------------------------------------------------
learnmet_tier2_cv <- function(
    pheno_all, geno_mat, env_cov, loc_coords, trait,
    n_folds = N_FOLDS,
    seed = RANDOM_STATE + TIER2_SEED_OFFSET,
    test_frac = TIER2_TEST_FRAC,
    val_frac = TIER2_VAL_FRAC,
    out_root = CV_OUT_ROOT
) {
  folds <- within_genotype_env_folds(
    genotypes = pheno_all$geno_ID,
    n_folds = n_folds,
    test_frac = test_frac,
    val_frac = val_frac,
    seed = seed
  )
  
  fold_metrics <- data.frame()
  pooled_true <- c()
  pooled_pred <- c()
  
  for (i in seq_along(folds)) {
    cat("\n", strrep("=", 80), "\n", sep="")
    cat(sprintf("Tier-2 Fold %d/%d (within-genotype env holdout)\n", i, n_folds))
    cat(strrep("=", 80), "\n", sep="")
    
    fold_tag <- sprintf("tier2_fold%02d", i)
    out <- run_learnmet_fold(
      pheno_all = pheno_all,
      geno_mat = geno_mat,
      env_cov = env_cov,
      loc_coords = loc_coords,
      trait = trait,
      train_idx = folds[[i]]$train,
      test_idx = folds[[i]]$test,
      fold_tag = fold_tag,
      out_root = out_root,
      seed = seed + i
    )
    
    m <- metrics(out$y_true, out$y_pred)
    
    fold_metrics <- bind_rows(
      fold_metrics,
      data.frame(fold=i, r2=m$r2, rmse=m$rmse, mae=m$mae, ccc=m$ccc)
    )
    
    pooled_true <- c(pooled_true, out$y_true)
    pooled_pred <- c(pooled_pred, out$y_pred)
    
    cat(sprintf("Fold %d metrics: R2=%.4f, RMSE=%.4f, MAE=%.4f, CCC=%.4f\n",
                i, m$r2, m$rmse, m$mae, m$ccc))
  }
  
  summary <- summarize_cv(fold_metrics)
  pooled  <- pooled_metrics_df(pooled_true, pooled_pred)
  
  dir.create(out_root, recursive=TRUE, showWarnings=FALSE)
  write.csv(fold_metrics, file.path(out_root, "learnmet_tier2_fold_metrics.csv"), row.names=FALSE)
  write.csv(summary,      file.path(out_root, "learnmet_tier2_summary.csv"), row.names=FALSE)
  write.csv(pooled,       file.path(out_root, "learnmet_tier2_pooled.csv"), row.names=FALSE)
  
  cat("\n", strrep("=", 80), "\n", sep="")
  cat("TIER-2 CV RESULTS (mean +/- sd)\n")
  cat(strrep("=", 80), "\n", sep="")
  print(summary)
  cat("\nPOOLED (micro) METRICS\n")
  print(pooled)
  
  invisible(list(fold_metrics=fold_metrics, summary=summary, pooled=pooled))
}


################################################################################
# 0) Paths / settings
################################################################################
DATA_DIR <- "/Users/thulanihewavithana/Documents/PhD/Research/EcoPopDL_GP/D4"
setwd(DATA_DIR)

GENO_PREFIX <- "imp.qc.all.withdc.clean.fixed"
PHENO_PATH  <- "dtf_mean.txt"
ENV_PATH    <- "d4_env_matrix_dtf.csv"
trait_name  <- "DTF"

# Feature-leakage flags (what you asked for)
YEAR_INCLUDED <- FALSE
LOCATION_INCLUDED <- FALSE
LATLON_INCLUDED <- FALSE
INCLUDE_ENV_PREDICTORS <- TRUE

################################################################################
# 1) Read input files
################################################################################
library(snpStats)

# --- 1a) Read PLINK -> geno_mat (PASTE YOUR EXACT PLINK BLOCK HERE) ---
# .bed and .fam stay the same
plinkData <- read.plink("imp.qc.all.withdc.clean.fixed.bed",
                        "imp.qc.all.withdc.clean.fixed.bim",
                        "imp.qc.all.withdc.clean.fixed.fam")

# 4. extract the SnpMatrix and coerce to a numeric matrix of 0/1/2
geno_mat <- as(plinkData$genotypes, "numeric")
# rows = samples, columns = SNPs

# --- 1b) Read phenotype -> pheno_all ---
pheno_all <- read.csv(PHENO_PATH, sep ='\t')

# --- 1c) Read env covariates -> env_cov ---
env_cov <- read.csv(ENV_PATH)

# --- 1d) Coordinates ---
locs <- unique(pheno_all$Location)  # or pheno_all$location aDTFer renaming
# build loc_coords like you do now (must cover all locations)
# loc_coords <- data.frame(location = locs,
#                          latitude  = c(50.3934, 50.9853),    # LL = Lucky Lake, MJ = Moose Jaw, YM = 32.6927
#                          longitude = c(-105.5520, -107.1330)) # YM = -114.6277
# loc_coords <- data.frame(location = locs,
#                          latitude  = c(23.6057, 57.1499,35.2010),    # fardipur, aberdeen, akransas
#                          longitude = c(89.8387, -2.0938,-91.8318)) 

# loc_coords <- data.frame(location = locs,
#                          latitude  = c(23.1208, 21.5222, 26.4499, 17.5287, 23.2032),    # [1] "Amlaha"      "Junagadh"    "Kanpur"      "Pathancheru" "Sehore"  
#                          longitude = c(76.9038, 70.4579, 80.3319, 78.2667, 77.0844))
loc_coords <- data.frame(location = locs,
                         latitude  = c(51.76635, 49.948968),    # [1] "Pike Lake" "Portage La Prairie" 
                         longitude = c(-106.85009, -98.231252))


loc_coords$location <- trimws(as.character(loc_coords$location))

################################################################################
# 2) Preprocess / rename columns (your existing code)
################################################################################
library(dplyr)


# 1) Make sure required column names exist
if (!"geno_ID" %in% names(pheno_all)) {
  if ("IID" %in% names(pheno_all)) {
    pheno_all <- pheno_all %>% rename(geno_ID = IID)
  } else if ("ID" %in% names(pheno_all)) {
    pheno_all <- pheno_all %>% rename(geno_ID = ID)
  } else {
    stop("Can't find IID or ID column to use as geno_ID.")
  }
}

if (!"location" %in% names(pheno_all)) {
  if ("Location" %in% names(pheno_all)) {
    pheno_all <- pheno_all %>% rename(location = Location)
  } else {
    stop("Can't find Location/location column.")
  }
}

if (!"year" %in% names(pheno_all)) {
  if ("Year" %in% names(pheno_all)) {
    pheno_all <- pheno_all %>% rename(year = Year)
  } else {
    stop("Can't find Year/year column.")
  }
}

# Optional but recommended for unique row keys:
# If you don't already have a 'Reps' column, use SD as the replicate identifier
if (!"Reps" %in% names(pheno_all)) {
  if ("SD" %in% names(pheno_all)) {
    pheno_all <- pheno_all %>% rename(Reps = SD)
  } else {
    pheno_all$Reps <- 1L
  }
}

# 2) Pick the trait column correctly (based on your file it is DTF)

stopifnot(trait_name %in% names(pheno_all))

# Clean strings
pheno_all$geno_ID  <- trimws(as.character(pheno_all$geno_ID))
pheno_all$location <- trimws(as.character(pheno_all$location))
pheno_all$year     <- as.numeric(pheno_all$year)

# Robust numeric conversion for DTF (handles factors + commas + spaces)
pheno_all$DTF <- as.numeric(gsub(",", "", trimws(as.character(pheno_all$DTF))))

cat("DTF class:", class(pheno_all$DTF), "\n")
cat("DTF NA count:", sum(is.na(pheno_all$DTF)), "out of", nrow(pheno_all), "\n")

# env feature columns
env_cols <- grep("^E_", names(env_cov), value = TRUE)

# 1 row per (year, location)
env_cov_env <- env_cov %>%
  mutate(
    year = as.numeric(year),
    location = trimws(as.character(location))
  ) %>%
  group_by(year, location) %>%
  summarize(across(all_of(env_cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop")

# sanity check: should equal number of unique year-location pairs
stopifnot(nrow(env_cov_env) == n_distinct(paste(env_cov_env$year, env_cov_env$location)))


# 1) Add stable row_id + split_key BEFORE splitting (used to match preds back to obs)
pheno_all <- pheno_all %>%
  mutate(
    geno_ID   = as.character(geno_ID),
    location  = as.character(location),
    year      = as.numeric(year),
    row_id    = dplyr::row_number(),
    split_key = if ("Reps" %in% names(.)) paste(geno_ID, location, year, Reps, sep="|")
    else paste(geno_ID, location, year, sep="|")
  )

# 2) Strongly recommended: filter to rows that have genotype + env info
pheno_all <- pheno_all %>%
  mutate(
    year = as.numeric(year),
    location = trimws(as.character(location)),
    geno_ID = trimws(as.character(geno_ID))
  ) %>%
  filter(geno_ID %in% rownames(geno_mat)) %>%
  semi_join(env_cov_env, by = c("year","location"))

pheno_all <- pheno_all %>% filter(!is.na(DTF))

# 3) Run Tier-1 (genotype-disjoint) CV
tier1_out <- learnmet_tier1_cv(
  pheno_all = pheno_all,
  geno_mat  = geno_mat,
  env_cov   = env_cov_env,   # <- IMPORTANT
  loc_coords= loc_coords,
  trait     = trait_name,
  n_folds   = N_FOLDS,
  seed      = RANDOM_STATE,
  out_root  = CV_OUT_ROOT
)

tier2_out <- learnmet_tier2_cv(
  pheno_all = pheno_all,
  geno_mat  = geno_mat,
  env_cov   = env_cov_env,   # <- IMPORTANT
  loc_coords= loc_coords,
  trait     = trait_name,
  n_folds   = N_FOLDS,
  seed      = RANDOM_STATE + TIER2_SEED_OFFSET,
  out_root  = CV_OUT_ROOT
)

