# Restores gitignored local artifacts from a backup snapshot.
param(
  [string]$Backup = "$env:TEMP\ice-final-snap",
  [string]$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $Backup)) {
  throw "Backup not found: $Backup — retrain instead (see RESTORE.md)"
}

Write-Host "Restoring from $Backup -> $Project"

New-Item -ItemType Directory -Force -Path (Join-Path $Project "data\clean") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Project "data\raw") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Project "models") | Out-Null

robocopy (Join-Path $Backup "data\clean") (Join-Path $Project "data\clean") /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
robocopy (Join-Path $Backup "data\raw") (Join-Path $Project "data\raw") /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null

$emb = Join-Path $Backup "models\embedding_classifier.joblib"
if (Test-Path $emb) {
  Copy-Item -Force $emb (Join-Path $Project "models\embedding_classifier.joblib")
}

$distil = Join-Path $Backup "models\distilbert"
if (Test-Path $distil) {
  robocopy $distil (Join-Path $Project "models\distilbert") /E /XD runs /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
}

Write-Host "Done. Run: python -m scripts.verify_artifacts"
