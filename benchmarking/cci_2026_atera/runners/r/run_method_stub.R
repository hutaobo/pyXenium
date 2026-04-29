args <- commandArgs(trailingOnly = TRUE)
payload <- list(
  status = "scaffolded",
  args = args,
  message = "This R method is registered and environment-managed, but its adapter is still a stub in this repo."
)
cat(jsonlite::toJSON(payload, pretty = TRUE, auto_unbox = TRUE))
