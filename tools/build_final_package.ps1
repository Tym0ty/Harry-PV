param(
    [string]$PackageBase = "final_package",
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$PackageName = "Harry-PV-final-github-$Stamp"
$PackageRoot = Join-Path $Repo $PackageBase
$PackageDir = Join-Path $PackageRoot $PackageName
$ManifestPath = Join-Path $PackageDir "PACKAGE_MANIFEST.csv"
$MissingPath = Join-Path $PackageDir "PACKAGE_MISSING.csv"

$Included = New-Object System.Collections.Generic.List[object]
$Missing = New-Object System.Collections.Generic.List[object]

function Convert-ToRepoRelative([string]$Path) {
    $full = (Resolve-Path -LiteralPath $Path).Path
    return $full.Substring($Repo.Length).TrimStart("\", "/")
}

function Add-Missing([string]$RelPath, [string]$Reason) {
    $Missing.Add([pscustomobject]@{
        path = $RelPath
        reason = $Reason
    })
}

function Copy-FileRel([string]$RelPath, [string]$Reason = "explicit") {
    $Src = Join-Path $Repo $RelPath
    if (-not (Test-Path -LiteralPath $Src -PathType Leaf)) {
        Add-Missing $RelPath $Reason
        return
    }
    $Dst = Join-Path $PackageDir $RelPath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Dst) | Out-Null
    Copy-Item -LiteralPath $Src -Destination $Dst -Force
    $Info = Get-Item -LiteralPath $Src
    $Included.Add([pscustomobject]@{
        path = $RelPath
        size_bytes = $Info.Length
        reason = $Reason
    })
}

function Copy-DirFiltered([string]$RelDir, [string[]]$Extensions, [string]$Reason) {
    $SrcDir = Join-Path $Repo $RelDir
    if (-not (Test-Path -LiteralPath $SrcDir -PathType Container)) {
        Add-Missing $RelDir $Reason
        return
    }
    Get-ChildItem -LiteralPath $SrcDir -Recurse -File -Force |
        Where-Object {
            $ext = $_.Extension.ToLowerInvariant()
            $Extensions -contains $ext -and
            $_.FullName -notmatch "\\__pycache__\\" -and
            $_.FullName -notmatch "\\.ipynb_checkpoints\\"
        } |
        ForEach-Object {
            $Rel = Convert-ToRepoRelative $_.FullName
            Copy-FileRel $Rel $Reason
        }
}

function Copy-TemplateAs([string]$TemplateRelPath, [string]$DestRelPath, [string]$Reason) {
    $Src = Join-Path $Repo (Join-Path "tools/package_templates" $TemplateRelPath)
    if (-not (Test-Path -LiteralPath $Src -PathType Leaf)) {
        Add-Missing "tools/package_templates/$TemplateRelPath" $Reason
        return
    }
    $Dst = Join-Path $PackageDir $DestRelPath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Dst) | Out-Null
    Copy-Item -LiteralPath $Src -Destination $Dst -Force
    $Info = Get-Item -LiteralPath $Src
    $Included.Add([pscustomobject]@{
        path = $DestRelPath
        size_bytes = $Info.Length
        reason = $Reason
    })
}

New-Item -ItemType Directory -Force -Path $PackageDir | Out-Null

$RootFiles = @(
    "requirements.txt",
    "run_all.py",
    "FINAL_FILE_SELECTION.md"
)

foreach ($file in $RootFiles) {
    Copy-FileRel $file "project-shell"
}

$PackageGitignore = @"
# Python and local environments
__pycache__/
*.pyc
*.pyo
*.egg-info/
.eggs/
dist/
build/
venv/
.venv/

# OS and IDE
.DS_Store
Thumbs.db
.vscode/
.idea/
.claude/

# Local secrets
.env
*.env

# Local generated package output
final_package/
release_package/
github_package/
*.zip
*.7z

# Regenerated bulky outputs that are not part of this curated final package
new_pipeline/data/output/experiments/**/raw/
new_pipeline/data/output/experiments/**/*raw*_M500*.parquet
new_pipeline/data/output/experiments/**/*hourly_dispatch*.parquet
new_pipeline/data/output/experiments/**/*_hourly.parquet
new_pipeline/data/output/experiments/**/*_daily.parquet
*.log

# Legacy handover/archive folders
BH_NEXT_CHAT_HANDOVER_20260606/
BH_THESIS_HANDOVER_2026_05_FINAL/
forecasting_handover_for_thesis_text/
THESIS_MAINLINE_ARCHIVE_2026_06/
"@

Set-Content -LiteralPath (Join-Path $PackageDir ".gitignore") -Value $PackageGitignore -Encoding UTF8
$PackageGitignoreInfo = Get-Item -LiteralPath (Join-Path $PackageDir ".gitignore")
$Included.Add([pscustomobject]@{
    path = ".gitignore"
    size_bytes = $PackageGitignoreInfo.Length
    reason = "package-gitignore"
})

Copy-TemplateAs ".gitattributes" ".gitattributes" "package-gitattributes"
Copy-TemplateAs "README.md" "README.md" "github-readme"
Copy-TemplateAs "LICENSE" "LICENSE" "license"
Copy-TemplateAs "CITATION.cff" "CITATION.cff" "citation"
Copy-TemplateAs "REPRODUCIBILITY.md" "REPRODUCIBILITY.md" "github-doc"
Copy-TemplateAs "PROJECT_STRUCTURE.md" "PROJECT_STRUCTURE.md" "github-doc"
Copy-FileRel "tools/build_final_package.ps1" "packaging-tool"
Copy-DirFiltered "tools/package_templates" @(".md", ".cff") "packaging-template"
Copy-FileRel "tools/package_templates/.gitattributes" "packaging-template"
Copy-FileRel "tools/package_templates/LICENSE" "packaging-template"

$FinalScripts = @(
    "new_pipeline/scripts/experiments/pv_focus_bridge_utils.py",
    "new_pipeline/scripts/experiments/exp_rebuild_pv_truth_from_cwa_ghi.py",
    "new_pipeline/scripts/experiments/exp_pv_focus_generate_pv_scenarios.py",
    "new_pipeline/scripts/experiments/exp_pv_focus_build_netload_scenarios.py",
    "new_pipeline/scripts/experiments/exp_pv_focus_reduce_scenarios.py",
    "new_pipeline/scripts/experiments/exp_pv_focus_lowpv_reduction_fix.py",
    "new_pipeline/scripts/experiments/exp_pv_focus_validate_scenarios.py",
    "new_pipeline/scripts/experiments/exp_pv_focus_reframed_validation.py",
    "new_pipeline/scripts/experiments/exp_pv_focus_tail_failure_attribution_audit.py",
    "new_pipeline/scripts/experiments/run_scheduling_pv_focus_da_mpc.py",
    "new_pipeline/scripts/experiments/run_scheduling_pv_focus_ccor_mpc.py",
    "new_pipeline/scripts/experiments/run_scheduling_pv_focus_ccor_v2_smoke.py",
    "new_pipeline/scripts/experiments/run_scheduling_pv_focus_ccor_v2_full_year_m150.py",
    "new_pipeline/scripts/experiments/run_final_four_load_bias_robustness_pvfocus.py",
    "new_pipeline/scripts/experiments/run_tailaware_ablation_pvfocus_km_vs_tkm.py",
    "new_pipeline/scripts/experiments/write_final_thesis_method_result_report_pvfocus.py",
    "new_pipeline/scripts/experiments/run_id_h24_ghi_prob_q19_forecast_only_v1.py",
    "new_pipeline/scripts/experiments/run_id_h24_ghi_prob_q19_independent_agaci_v4.py"
)

foreach ($file in $FinalScripts) {
    Copy-FileRel $file "final-code"
}

$MilpFiles = @(
    "milp_v2/__init__.py",
    "milp_v2/common.py",
    "milp_v2/config.yaml"
)

foreach ($file in $MilpFiles) {
    Copy-FileRel $file "milp-core"
}

Copy-DirFiltered "milp_v2/layer_a" @(".py", ".md", ".yaml", ".yml", ".json", ".csv") "milp-layer-a"
Copy-DirFiltered "milp_v2/layer_b" @(".py", ".md", ".yaml", ".yml", ".json", ".csv") "milp-layer-b"

Copy-DirFiltered "new_pipeline/data/output/experiments/final_thesis_method_result_report_pvfocus" @(".md", ".csv", ".json", ".txt") "final-report"

$EvidenceFiles = @(
    "new_pipeline/data/output/experiments/scheduling_pv_focus_lowpv_safe_da_mpc/annual_cost_summary_realized_basis.csv",
    "new_pipeline/data/output/experiments/scheduling_pv_focus_ccor_v2_full_year_m150/annual_cost_summary_realized_basis.csv",
    "new_pipeline/data/output/experiments/final_four_load_bias_robustness_pvfocus/annual_cost_summary_realized_basis.csv",
    "new_pipeline/data/output/experiments/tailaware_ablation_pvfocus_km_vs_tkm/annual_cost_summary_realized_basis.csv",
    "new_pipeline/data/output/experiments/tailaware_ablation_pvfocus_km_vs_tkm/km_vs_tkm_delta_comparison.csv",
    "new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild/cwa_pv_truth_summary.csv",
    "new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild/full_year_replay_truth_package_CWA_GHI_PV.parquet",
    "new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/load_bias_profiles.parquet",
    "new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet",
    "new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/PV_FOCUSED_VALIDATION_REFRAMED_LOW_PV_FIX_REPORT.md",
    "new_pipeline/data/output/experiments/id_h24_ghi_prob/id_h24_ghi_prob_metrics.csv",
    "new_pipeline/data/output/experiments/id_h24_ghi_prob/id_h24_pv_prob_metrics.csv",
    "new_pipeline/data/output/experiments/main_forecasting_model_comparison/main_model_comparison.csv",
    "new_pipeline/data/output/experiments/lit_guided_id_h24_ghi/literature_guided_id_h24_ghi_model_comparison.csv",
    "new_pipeline/data/output/experiments/figures/table1_agaci_metrics.csv"
)

foreach ($file in $EvidenceFiles) {
    Copy-FileRel $file "final-evidence"
}

$EvidenceDirs = @(
    "new_pipeline/data/output/experiments/scheduling_pv_focus_lowpv_safe_da_mpc",
    "new_pipeline/data/output/experiments/scheduling_pv_focus_ccor_v2_full_year_m150",
    "new_pipeline/data/output/experiments/final_four_load_bias_robustness_pvfocus",
    "new_pipeline/data/output/experiments/tailaware_ablation_pvfocus_km_vs_tkm",
    "new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild",
    "new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge",
    "new_pipeline/data/output/experiments/id_h24_ghi_prob",
    "new_pipeline/data/output/experiments/main_forecasting_model_comparison",
    "new_pipeline/data/output/experiments/lit_guided_id_h24_ghi",
    "new_pipeline/data/output/experiments/figures"
)

foreach ($dir in $EvidenceDirs) {
    Copy-DirFiltered $dir @(".md", ".csv", ".json", ".txt", ".png") "compact-final-evidence"
}

Copy-DirFiltered "thesis_figures" @(".png", ".csv", ".md", ".txt") "thesis-assets"
Copy-DirFiltered "thesis_tables" @(".csv", ".md", ".txt") "thesis-assets"
Copy-FileRel "FIGURES_TABLES_INDEX.md" "thesis-assets"

$OptionalThesisDirs = @(
    "new_pipeline/data/output/thesis_ch4_forecast_figures_v1",
    "new_pipeline/data/output/thesis_schedule_visualization_prob_ccor_m150_v2",
    "new_pipeline/data/output/thesis_mechanism_figures_48h",
    "new_pipeline/data/output/thesis_table_4_3_panel_b_v1",
    "new_pipeline/data/output/thesis_table_4_4_six_case_bar_figure_v1",
    "new_pipeline/data/output/oral_slide_ghi_quantile_calibration_v2"
)

foreach ($dir in $OptionalThesisDirs) {
    Copy-DirFiltered $dir @(".md", ".csv", ".json", ".txt", ".png") "optional-thesis-figure-package"
}

$ReferenceDocs = @(
    "RESEARCH_PIPELINE_SUMMARY.md",
    "DECISION_LOG.md",
    "EXPERIMENT_PROGRESS_AND_METHOD_LOG.md",
    "SCENARIO_GENERATION_TCOPULA_AUDIT.md",
    "FORECASTING_DA_ID_MODEL_COMPARISON_FINAL.md",
    "FINAL_METHOD_LOGIC_AUDIT_AND_RERUN_RECOMMENDATION.md",
    "DAY_AHEAD_INTRADAY_MILP_MPC_MODEL_AUDIT.md",
    "DA_AWARE_MPC_RERUN_RESULTS_AND_METHOD_DECISION.md",
    "DA_AWARE_MPC_VS_DA_MODEL_DIFFERENCE_AUDIT.md"
)

foreach ($file in $ReferenceDocs) {
    Copy-FileRel $file "reference-doc"
}

$Readme = @"
# Harry-PV Final GitHub Package

This package was generated by tools/build_final_package.ps1.

Primary final line:

- PV-focused rolling MPC / CCOR-MPC

Primary final report:

- new_pipeline/data/output/experiments/final_thesis_method_result_report_pvfocus/FINAL_THESIS_METHOD_AND_RESULTS_REPORT_PVFOCUS.md

Final main cases:

- MPC_DET
- MPC_PROB
- CCOR_DET
- CCOR_PROB

Notes:

- Large raw outputs, old handover archives, virtual environments, logs, and regenerated intermediate files are excluded.
- PACKAGE_MANIFEST.csv lists included files.
- PACKAGE_MISSING.csv lists expected files that were not found, if any.
- FINAL_FILE_SELECTION.md explains the inclusion/exclusion judgement.
"@

Set-Content -LiteralPath (Join-Path $PackageDir "PACKAGE_README.md") -Value $Readme -Encoding UTF8
$Info = Get-Item -LiteralPath (Join-Path $PackageDir "PACKAGE_README.md")
$Included.Add([pscustomobject]@{
    path = "PACKAGE_README.md"
    size_bytes = $Info.Length
    reason = "package-readme"
})

$Included |
    Sort-Object path -Unique |
    Export-Csv -LiteralPath $ManifestPath -NoTypeInformation -Encoding UTF8

$Missing |
    Sort-Object path -Unique |
    Export-Csv -LiteralPath $MissingPath -NoTypeInformation -Encoding UTF8

if (-not $NoZip) {
    $ZipPath = "$PackageDir.zip"
    if (Test-Path -LiteralPath $ZipPath) {
        Remove-Item -LiteralPath $ZipPath -Force
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $Zip = [System.IO.Compression.ZipFile]::Open($ZipPath, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        Get-ChildItem -LiteralPath $PackageDir -Recurse -File -Force |
            Where-Object { $_.FullName -notmatch "\\.git\\" } |
            ForEach-Object {
                $EntryName = $_.FullName.Substring($PackageDir.Length).TrimStart("\", "/")
                [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                    $Zip,
                    $_.FullName,
                    $EntryName,
                    [System.IO.Compression.CompressionLevel]::Optimal
                ) | Out-Null
            }
    } finally {
        $Zip.Dispose()
    }
    $ZipInfo = Get-Item -LiteralPath $ZipPath
    Write-Host "ZIP=$ZipPath"
    Write-Host ("ZIP_SIZE_MB={0:N2}" -f ($ZipInfo.Length / 1MB))
}

$FileCount = (Get-ChildItem -LiteralPath $PackageDir -Recurse -File | Measure-Object).Count
$TotalBytes = (Get-ChildItem -LiteralPath $PackageDir -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host "PACKAGE_DIR=$PackageDir"
Write-Host "FILE_COUNT=$FileCount"
Write-Host ("PACKAGE_SIZE_MB={0:N2}" -f ($TotalBytes / 1MB))
Write-Host "MANIFEST=$ManifestPath"
Write-Host "MISSING=$MissingPath"
