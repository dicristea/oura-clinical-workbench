#!/usr/bin/env Rscript

# Convert the de-identified hepatology RDS file into dashboard-friendly JSON.
# Run from any directory:
#
#   Rscript di_data/convert_hepatology_rds.R
#
# Optional positional arguments:
#   1. input RDS path
#   2. output directory
#   3. number of participants, or "all" (default: 10)
#   4. within-day records: integer, "per-participant", "all", or "none"
#      (default: per-participant)
#
# For stable pseudonyms across refreshed source files, optionally set a private
# HEPATOLOGY_ID_SALT environment variable before running the script.

suppressPackageStartupMessages({
  library(dplyr)
  library(digest)
  library(haven)
  library(jsonlite)
})

script_args <- commandArgs(trailingOnly = FALSE)
script_file_arg <- grep("^--file=", script_args, value = TRUE)
script_dir <- if (length(script_file_arg) == 1) {
  dirname(normalizePath(sub("^--file=", "", script_file_arg)))
} else {
  normalizePath("di_data")
}

args <- commandArgs(trailingOnly = TRUE)
default_input <- file.path(script_dir, "de-iden_data_Adam_2026-09-22.rds")
default_output <- file.path(script_dir, "exports", "hepatology_sample_10")

input_path <- if (length(args) >= 1) args[[1]] else default_input
output_dir <- if (length(args) >= 2) args[[2]] else default_output
participant_count_arg <- if (length(args) >= 3) tolower(args[[3]]) else "10"
within_day_arg <- if (length(args) >= 4) tolower(args[[4]]) else "per-participant"

export_all_participants <- identical(participant_count_arg, "all")
participant_count <- suppressWarnings(as.integer(participant_count_arg))
if (!export_all_participants && (is.na(participant_count) || participant_count < 1)) {
  stop("participant_count must be a positive integer or 'all'.")
}

valid_within_day_modes <- c("per-participant", "all", "none")
within_day_count <- suppressWarnings(as.integer(within_day_arg))
if (!within_day_arg %in% valid_within_day_modes &&
    (is.na(within_day_count) || within_day_count < 0)) {
  stop(
    "within-day argument must be a non-negative integer, ",
    "'per-participant', 'all', or 'none'."
  )
}
if (!file.exists(input_path)) {
  stop("Input RDS file not found: ", input_path)
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

hepatology_data <- readRDS(input_path)

participant_fields <- c(
  "name", "sex", "age_yrs", "decomp", "he_meds", "ctp_class",
  "psqiglobal", "cldq_fatigue", "cldq_activity", "cldq_total",
  "che_phes", "ever_he_meds", "meld", "cirrhosis_etiology",
  "medication_for_ascites", "variceal_bleeding_history",
  "liver_frailty_index", "phes_score", "ctp_cat", "meld_cat",
  "cldq_total_cat", "cldq_activity_cat", "cldq_fatigue_cat",
  "lfi_cat", "edu_cat", "race_cat", "race_eth_cat", "anyhe",
  "anyOHE", "robust", "frail", "ascites"
)

participant_day_fields <- c(
  "days_since_start", "day_type", "average_heart_rate", "average_hrv",
  "average_breath", "spo2", "average_met_minutes", "steps", "log_steps",
  "non_wear_time", "total_calories", "num_met", "num_class_5_min",
  "valid80", "non_wear_pct"
)

within_day_fields <- c("class_5_min", "met_val", "class5_val", "class1_val")
expected_fields <- c(participant_fields, participant_day_fields, within_day_fields)

missing_fields <- setdiff(expected_fields, names(hepatology_data))
unexpected_fields <- setdiff(names(hepatology_data), expected_fields)
if (length(missing_fields) > 0) {
  stop("RDS is missing expected fields: ", paste(missing_fields, collapse = ", "))
}
if (length(unexpected_fields) > 0) {
  warning("RDS contains unexpected fields: ", paste(unexpected_fields, collapse = ", "))
}
if (length(names(hepatology_data)) != 51) {
  stop("Expected 51 source fields, found ", length(names(hepatology_data)), ".")
}

source_sha256 <- digest::digest(file = input_path, algo = "sha256")

hepatology_data <- hepatology_data |>
  mutate(.raw_participant_id = as.character(name))

if (any(is.na(hepatology_data$.raw_participant_id) | hepatology_data$.raw_participant_id == "")) {
  stop("Every row must have a non-empty participant identifier in `name`.")
}

duplicate_days <- hepatology_data |>
  count(.raw_participant_id, days_since_start, name = "row_count") |>
  filter(row_count > 1)
if (nrow(duplicate_days) > 0) {
  stop("Found duplicate participant-day rows; resolve them before conversion.")
}

# For a useful demo, choose participants with the most valid longitudinal data.
coverage <- hepatology_data |>
  group_by(.raw_participant_id) |>
  summarise(
    total_days = n(),
    valid_days = sum(valid80 == 1, na.rm = TRUE),
    first_study_day = min(days_since_start, na.rm = TRUE),
    last_study_day = max(days_since_start, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(desc(valid_days), desc(total_days), .raw_participant_id)

participant_count <- if (export_all_participants) {
  nrow(coverage)
} else {
  min(participant_count, nrow(coverage))
}
selected_coverage <- coverage |> slice_head(n = participant_count)
selected_raw_ids <- selected_coverage$.raw_participant_id

id_salt <- Sys.getenv("HEPATOLOGY_ID_SALT", unset = "")
uses_custom_id_salt <- nzchar(id_salt)
if (!uses_custom_id_salt) {
  id_salt <- "hepatology-dashboard-local-v1"
  warning(
    "HEPATOLOGY_ID_SALT is not set; using the local demo salt. ",
    "Set a private salt before producing a long-lived shared dataset."
  )
}
anonymous_ids <- vapply(selected_raw_ids, function(raw_id) {
  token <- digest::hmac(
    key = id_salt,
    object = raw_id,
    algo = "sha256",
    serialize = FALSE
  )
  paste0("HEP-", toupper(substr(token, 1, 12)))
}, character(1))
if (anyDuplicated(anonymous_ids)) {
  stop("Anonymous ID collision detected; increase the exported hash length.")
}
id_map <- setNames(anonymous_ids, selected_raw_ids)

scalar_json_value <- function(column, index) {
  value <- column[index]

  if (length(value) == 0 || is.na(value[[1]])) {
    if (is.numeric(column) || is.integer(column)) return(NA_real_)
    if (is.logical(column)) return(NA)
    return(NA_character_)
  }

  if (inherits(column, "haven_labelled")) {
    return(as.character(haven::as_factor(value, levels = "default"))[[1]])
  }
  if (is.factor(column)) return(as.character(value)[[1]])
  if (inherits(column, "Date")) return(format(value[[1]], "%Y-%m-%d"))
  if (is.integer(column)) return(as.integer(value[[1]]))
  if (is.numeric(column)) return(as.numeric(value[[1]]))
  if (is.logical(column)) return(as.logical(value[[1]]))
  as.character(value[[1]])
}

first_non_missing_index <- function(column, candidate_indices) {
  available <- candidate_indices[!is.na(column[candidate_indices])]
  if (length(available) > 0) available[[1]] else candidate_indices[[1]]
}

participants <- lapply(selected_raw_ids, function(raw_id) {
  row_indices <- which(hepatology_data$.raw_participant_id == raw_id)
  static_features <- setNames(
    lapply(setdiff(participant_fields, "name"), function(field) {
      index <- first_non_missing_index(hepatology_data[[field]], row_indices)
      scalar_json_value(hepatology_data[[field]], index)
    }),
    setdiff(participant_fields, "name")
  )
  participant_coverage <- selected_coverage |>
    filter(.raw_participant_id == raw_id) |>
    slice_head(n = 1)

  list(
    patient_id = unname(id_map[[raw_id]]),
    data_source = "hepatology_deidentified_rds",
    static_features = static_features,
    coverage = list(
      total_days = as.integer(participant_coverage$total_days[[1]]),
      valid_days = as.integer(participant_coverage$valid_days[[1]]),
      first_study_day = as.integer(participant_coverage$first_study_day[[1]]),
      last_study_day = as.integer(participant_coverage$last_study_day[[1]])
    )
  )
})

selected_indices <- which(hepatology_data$.raw_participant_id %in% selected_raw_ids)
selected_indices <- selected_indices[
  order(
    match(hepatology_data$.raw_participant_id[selected_indices], selected_raw_ids),
    hepatology_data$days_since_start[selected_indices]
  )
]

daily_records <- lapply(selected_indices, function(index) {
  raw_id <- hepatology_data$.raw_participant_id[[index]]
  record <- list(
    patient_id = unname(id_map[[raw_id]]),
    study_day = as.integer(hepatology_data$days_since_start[[index]])
  )
  for (field in setdiff(participant_day_fields, "days_since_start")) {
    record[[field]] <- scalar_json_value(hepatology_data[[field]], index)
  }
  record
})

# Prefer one valid day from each selected participant for a compact within-day sample.
primary_within_indices <- unlist(lapply(selected_raw_ids, function(raw_id) {
  candidates <- which(
    hepatology_data$.raw_participant_id == raw_id &
      !is.na(hepatology_data$valid80) & hepatology_data$valid80 == 1
  )
  if (length(candidates) > 0) candidates[[1]] else integer(0)
}), use.names = FALSE)

remaining_valid_indices <- setdiff(
  selected_indices[
    !is.na(hepatology_data$valid80[selected_indices]) &
      hepatology_data$valid80[selected_indices] == 1
  ],
  primary_within_indices
)
within_indices <- if (within_day_arg == "all") {
  selected_indices
} else if (within_day_arg == "none" || (!is.na(within_day_count) && within_day_count == 0)) {
  integer(0)
} else {
  requested_count <- if (within_day_arg == "per-participant") {
    length(selected_raw_ids)
  } else {
    within_day_count
  }
  head(c(primary_within_indices, remaining_valid_indices), requested_count)
}

within_vector <- function(field, index) {
  if (field == "class_5_min") {
    value <- hepatology_data[[field]][[index]]
    if (is.na(value) || value == "") return(integer(0))
    return(as.integer(strsplit(value, "", fixed = TRUE)[[1]]))
  }

  value <- hepatology_data[[field]][[index]]
  if (field %in% c("class5_val", "class1_val")) {
    return(as.integer(value))
  }
  as.numeric(value)
}

within_day_records <- lapply(within_indices, function(index) {
  raw_id <- hepatology_data$.raw_participant_id[[index]]
  series <- setNames(
    lapply(within_day_fields, function(field) within_vector(field, index)),
    within_day_fields
  )
  list(
    patient_id = unname(id_map[[raw_id]]),
    study_day = as.integer(hepatology_data$days_since_start[[index]]),
    series_lengths = lapply(series, length),
    series = series
  )
})

field_level <- function(field) {
  if (field %in% participant_fields) return("participant")
  if (field %in% participant_day_fields) return("participant_day")
  if (field %in% within_day_fields) return("within_day")
  NA_character_
}

column_missing_percent <- function(column) {
  missing <- if (is.list(column)) {
    vapply(column, function(value) {
      length(value) == 0 || all(is.na(value))
    }, logical(1))
  } else {
    is.na(column)
  }
  round(mean(missing) * 100, 1)
}

variable_dictionary <- lapply(names(hepatology_data)[names(hepatology_data) != ".raw_participant_id"], function(field) {
  column <- hepatology_data[[field]]
  variable_label <- attr(column, "label")
  value_labels <- attr(column, "labels")

  list(
    variable = field,
    level = field_level(field),
    r_class = class(column)[[1]],
    variable_label = if (is.null(variable_label)) "" else as.character(variable_label),
    value_labels = if (is.null(value_labels)) {
      list()
    } else {
      as.list(setNames(as.vector(value_labels), names(value_labels)))
    },
    missing_percent = column_missing_percent(column)
  )
})

comparable_values <- function(column) {
  if (inherits(column, "haven_labelled")) {
    return(as.character(haven::as_factor(column, levels = "default")))
  }
  as.character(column)
}

static_inconsistencies <- lapply(setdiff(participant_fields, "name"), function(field) {
  values <- comparable_values(hepatology_data[[field]])
  by_participant <- split(values, hepatology_data$.raw_participant_id)
  inconsistent_count <- sum(vapply(by_participant, function(group_values) {
    length(unique(group_values[!is.na(group_values)])) > 1
  }, logical(1)))
  list(variable = field, inconsistent_participants = inconsistent_count)
})
static_inconsistencies <- Filter(
  function(item) item$inconsistent_participants > 0,
  static_inconsistencies
)

sequence_validation <- list(
  class_5_min_invalid_length = sum(nchar(hepatology_data$class_5_min) != 288, na.rm = TRUE),
  met_val_invalid_length = sum(lengths(hepatology_data$met_val) != 1440),
  class5_val_invalid_length = sum(lengths(hepatology_data$class5_val) != 288),
  class1_val_invalid_length = sum(lengths(hepatology_data$class1_val) != 1440)
)
unexpected_valid80 <- sort(unique(hepatology_data$valid80[!is.na(hepatology_data$valid80)]))
unexpected_valid80 <- unexpected_valid80[!unexpected_valid80 %in% c(0, 1)]

validation_issue_count <- length(static_inconsistencies) +
  sum(unlist(sequence_validation)) +
  length(unexpected_valid80)

validation_report <- list(
  status = if (validation_issue_count == 0) "passed" else "passed_with_warnings",
  checks = list(
    source_field_count = list(
      expected = 51L,
      observed = length(setdiff(names(hepatology_data), ".raw_participant_id")),
      passed = length(setdiff(names(hepatology_data), ".raw_participant_id")) == 51L
    ),
    duplicate_participant_days = list(count = nrow(duplicate_days), passed = nrow(duplicate_days) == 0),
    static_field_inconsistencies = static_inconsistencies,
    within_day_sequence_lengths = sequence_validation,
    unexpected_valid80_values = as.list(unexpected_valid80)
  )
)

manifest <- list(
  schema_version = "1.1",
  source_file = basename(input_path),
  source_sha256 = source_sha256,
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  source_summary = list(
    participant_count = length(unique(hepatology_data$.raw_participant_id)),
    participant_day_count = nrow(hepatology_data),
    variable_count = 51L
  ),
  sample_summary = list(
    selection = if (export_all_participants) {
      "All participants"
    } else {
      "Participants with the most valid80 days, then the most total days"
    },
    participant_count = length(participants),
    participant_day_count = length(daily_records),
    within_day_example_count = length(within_day_records),
    patient_ids = anonymous_ids,
    within_day_mode = within_day_arg
  ),
  identity = list(
    strategy = "HMAC-SHA256 truncated to 12 hexadecimal characters",
    custom_salt_used = uses_custom_id_salt,
    raw_identifiers_exported = FALSE
  ),
  field_groups = list(
    participant = setdiff(participant_fields, "name"),
    participant_day = c("study_day", setdiff(participant_day_fields, "days_since_start")),
    within_day = within_day_fields
  ),
  notes = c(
    "Original participant identifiers are not exported.",
    "study_day is relative because the source file does not contain calendar dates.",
    "daily_records.json contains all available days for the selected participants.",
    "within_day_records.json follows the requested export mode and retains full arrays.",
    "class_5_min is normalized from a 288-character string to an integer array."
  )
)

write_json_file <- function(object, filename) {
  path <- file.path(output_dir, filename)
  jsonlite::write_json(
    object,
    path,
    pretty = TRUE,
    auto_unbox = TRUE,
    na = "null",
    null = "null",
    digits = NA
  )
  path
}

output_paths <- c(
  write_json_file(manifest, "manifest.json"),
  write_json_file(participants, "participants.json"),
  write_json_file(daily_records, "daily_records.json"),
  write_json_file(within_day_records, "within_day_records.json"),
  write_json_file(variable_dictionary, "variable_dictionary.json"),
  write_json_file(validation_report, "validation_report.json")
)

cat("Hepatology RDS conversion complete.\n")
cat("  Source participants:", manifest$source_summary$participant_count, "\n")
cat("  Source participant-days:", manifest$source_summary$participant_day_count, "\n")
cat("  Exported participants:", manifest$sample_summary$participant_count, "\n")
cat("  Exported participant-days:", manifest$sample_summary$participant_day_count, "\n")
cat("  Exported within-day examples:", manifest$sample_summary$within_day_example_count, "\n")
cat("  Validation status:", validation_report$status, "\n")
cat("  Output directory:", normalizePath(output_dir), "\n")
cat("  Files:\n")
cat(paste0("    - ", basename(output_paths), collapse = "\n"), "\n")
