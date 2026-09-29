[CmdletBinding()]
param(
    [string]$LogDir,
    [ValidateRange(1, 300)]
    [int]$RefreshSeconds = 5,
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($LogDir)) {
    $operationsRoot = Join-Path $repositoryRoot 'reports\operations'
    $latest = Get-ChildItem -LiteralPath $operationsRoot -Directory |
        Where-Object {
            (Test-Path -LiteralPath (Join-Path $_.FullName 'metrics.jsonl')) -and
            (Test-Path -LiteralPath (Join-Path $_.FullName 'stderr.log'))
        } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        throw "No monitored study was found below $operationsRoot"
    }
    $resolvedLogDir = $latest.FullName
}
else {
    $candidate = if ([IO.Path]::IsPathRooted($LogDir)) {
        $LogDir
    }
    else {
        Join-Path $repositoryRoot $LogDir
    }
    $resolvedLogDir = [IO.Path]::GetFullPath($candidate)
}

$metricsPath = Join-Path $resolvedLogDir 'metrics.jsonl'
$stderrPath = Join-Path $resolvedLogDir 'stderr.log'
$eventsPath = Join-Path $resolvedLogDir 'events.jsonl'
if (-not (Test-Path -LiteralPath $metricsPath) -or
    -not (Test-Path -LiteralPath $stderrPath)) {
    throw "The monitoring directory must contain metrics.jsonl and stderr.log: $resolvedLogDir"
}

$banner = @'
 _____ ____      _    ____ ___ _   _  ____
|_   _|  _ \    / \  |  _ \_ _| \ | |/ ___|
  | | | |_) |  / _ \ | | | | ||  \| | |  _
  | | |  _ <  / ___ \| |_| | || |\  | |_| |
  |_| |_| \_\/_/   \_\____/___|_| \_|\____|

       EXPANDED RESEARCH STUDY :: LIVE MONITOR
'@

function Clear-MonitorScreen {
    if (-not [Console]::IsOutputRedirected) {
        try {
            [Console]::Clear()
        }
        catch {
            # Some embedded PowerShell hosts expose a console without a writable
            # screen buffer. Rendering remains useful without clearing it.
        }
    }
}

function Get-LatestEventLine {
    param([string]$Path)

    $lines = @(Get-Content -LiteralPath $Path -Tail 500 -ErrorAction Stop)
    return $lines | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        Select-Object -Last 1
}

function Get-RunContext {
    param(
        [string]$Path,
        [string]$EventsPath
    )

    $lines = @(Get-Content -LiteralPath $Path -ErrorAction Stop)
    $startLine = $lines |
        Where-Object { $_ -match 'Expanded study started run_id=' } |
        Select-Object -Last 1
    $runId = if ($null -ne $startLine -and $startLine -match 'run_id=(?<id>\S+)') {
        $Matches['id']
    } else { $null }
    $runDirectory = if ($null -ne $runId) {
        Join-Path $repositoryRoot "runs\expanded_closeout\$runId"
    } else { $null }

    $phase = 'INITIALISING'
    $activity = 'Starting study'
    if ($lines -match 'Controlled sensitivity started') {
        $phase = 'HPO'
        $activity = 'Controlled sensitivity'
    }
    if ($lines -match 'Controlled sensitivity completed') {
        $phase = 'HPO'
        $activity = 'Nested candidate selection'
        if ($lines[-1] -match 'Outer evaluation started') {
            $activity = 'Outer-fold evaluation'
        }
        elseif ($lines[-1] -match 'RL arm started') {
            $activity = 'RL candidate selection'
        }
    }
    if ($lines -match 'Family locks sealed') {
        $phase = 'FINAL TEST'
        $activity = 'Final training and descriptive holdout'
        if ($lines[-1] -match 'Descriptive RL holdout fit started') {
            $activity = 'Descriptive RL holdout'
        }
        elseif ($lines[-1] -match 'Descriptive holdout fit started') {
            $activity = 'Descriptive supervised holdout'
        }
    }
    if ($lines -match 'Expanded study completed') {
        $phase = 'COMPLETED'
        $activity = 'Study complete'
    }
    $operationalWarning = $null
    $lastOperationalEvent = $null
    if (Test-Path -LiteralPath $EventsPath) {
        $lastOperationalEvent = Get-Content -LiteralPath $EventsPath -Tail 1 |
            ConvertFrom-Json
        if ($lastOperationalEvent.event -in @('failed', 'supervisor_failure')) {
            $phase = 'FAILED'
            $activity = 'Study stopped with an error'
        }
        elseif ($lastOperationalEvent.event -eq 'finished') {
            $phase = 'COMPLETED'
            $activity = 'Study complete'
        }
    }
    $completionVerified = $false
    if ($null -ne $runDirectory -and (Test-Path -LiteralPath $runDirectory)) {
        $completionPath = Join-Path $runDirectory 'completion.json'
        $auditPath = Join-Path $runDirectory 'audit.json'
        if ((Test-Path -LiteralPath $completionPath) -and
            (Test-Path -LiteralPath $auditPath)) {
            $completion = Get-Content -Raw -LiteralPath $completionPath | ConvertFrom-Json
            $audit = Get-Content -Raw -LiteralPath $auditPath | ConvertFrom-Json
            $auditHash = (Get-FileHash -LiteralPath $auditPath -Algorithm SHA256).Hash.ToLower()
            $completionVerified = $completion.status -eq 'complete' -and
                $audit.status -eq 'passed' -and
                $completion.run_id -eq $runId -and
                $auditHash -eq ([string]$completion.audit_sha256).ToLower()
        }
    }
    if ($completionVerified) {
        $phase = 'COMPLETED'
        $activity = 'Study complete; immutable audit passed'
        if ($null -ne $lastOperationalEvent -and
            $lastOperationalEvent.event -in @('failed', 'supervisor_failure')) {
            $operationalWarning = 'Post-completion process exit anomaly: supervisor exit code {0}' -f
                $lastOperationalEvent.exit_code
        }
    }
    $holdoutStartedArms = @($lines | ForEach-Object {
        if ($_ -match 'Descriptive (?:RL )?holdout fit started arm=(?<arm>\S+)') {
            $Matches['arm']
        }
    } | Select-Object -Unique)

    return [pscustomobject]@{
        RunId = $runId
        RunDirectory = $runDirectory
        Phase = $phase
        Activity = $activity
        CompletionVerified = $completionVerified
        OperationalWarning = $operationalWarning
        SensitivityActive = -not ($lines -match 'Controlled sensitivity completed')
        HoldoutStartedArms = $holdoutStartedArms
    }
}

function Format-Duration {
    param([double]$Seconds)

    if ($Seconds -le 0) { return '-' }
    $span = [TimeSpan]::FromSeconds($Seconds)
    if ($span.TotalDays -ge 1) {
        return '{0}d {1:00}:{2:00}:{3:00}' -f [math]::Floor($span.TotalDays),
            $span.Hours, $span.Minutes, $span.Seconds
    }
    return '{0:00}:{1:00}:{2:00}' -f [math]::Floor($span.TotalHours),
        $span.Minutes, $span.Seconds
}

function Format-RoundedEstimate {
    param([double]$Seconds)

    $roundedSeconds = Get-RoundedEstimateSeconds $Seconds
    $span = [TimeSpan]::FromSeconds($roundedSeconds)
    if ($span.TotalDays -ge 1) {
        return '{0}d {1}h {2}m' -f [math]::Floor($span.TotalDays),
            $span.Hours, $span.Minutes
    }
    if ($span.TotalHours -ge 1) {
        return '{0}h {1}m' -f [math]::Floor($span.TotalHours), $span.Minutes
    }
    return '{0}m' -f $span.Minutes
}

function Get-RoundedEstimateSeconds {
    param([double]$Seconds)

    return [math]::Max(0, [math]::Round($Seconds / 300) * 300)
}

function Get-ArmProgress {
    param([pscustomobject]$Context)

    if ([string]::IsNullOrWhiteSpace($Context.RunDirectory) -or
        -not (Test-Path -LiteralPath $Context.RunDirectory)) {
        return $null
    }

    $protocolPath = Join-Path $Context.RunDirectory 'protocol.json'
    $designPath = Join-Path $Context.RunDirectory 'sensitivity_design.json'
    $ledgerName = if ($Context.SensitivityActive) {
        'sensitivity_trial_ledger.jsonl'
    } else { 'trial_ledger.jsonl' }
    $ledgerPath = Join-Path $Context.RunDirectory $ledgerName
    if (-not (Test-Path -LiteralPath $protocolPath) -or
        -not (Test-Path -LiteralPath $ledgerPath)) {
        return $null
    }

    $protocol = Get-Content -Raw -LiteralPath $protocolPath | ConvertFrom-Json
    $calibration = $null
    $capabilityGatePath = Join-Path $repositoryRoot $protocol.authority.capability_gate
    if (Test-Path -LiteralPath $capabilityGatePath) {
        $gateHash = (Get-FileHash -LiteralPath $capabilityGatePath -Algorithm SHA256).Hash.ToLower()
        if ($gateHash -eq ([string]$protocol.authority.capability_gate_sha256).ToLower()) {
            $gate = Get-Content -Raw -LiteralPath $capabilityGatePath | ConvertFrom-Json
            $calibrationPath = Join-Path $repositoryRoot $gate.calibration_evidence.path
            if (Test-Path -LiteralPath $calibrationPath) {
                $calibrationHash = (Get-FileHash -LiteralPath $calibrationPath `
                    -Algorithm SHA256).Hash.ToLower()
                if ($calibrationHash -eq ([string]$gate.calibration_evidence.sha256).ToLower()) {
                    $calibration = Get-Content -Raw -LiteralPath $calibrationPath |
                        ConvertFrom-Json
                }
            }
        }
    }
    $outerCount = [int]$protocol.validation.window_policy.outer_folds
    $folds = @(1..$outerCount | ForEach-Object { "outer_$_" })
    $riskScenarioCount = @($protocol.portfolio.risk_scenarios.PSObject.Properties).Count
    $arms = @($protocol.experiment_arms)
    $seedCount = @($protocol.reproducibility.seeds).Count
    $design = if (Test-Path -LiteralPath $designPath) {
        Get-Content -Raw -LiteralPath $designPath | ConvertFrom-Json
    } else { $null }
    $innerCount = if ($Context.SensitivityActive -and $null -ne $design) {
        [int]$design.inner_folds_per_outer
    } else { [int]$protocol.validation.window_policy.inner_folds }
    $stages = @()
    foreach ($outerNumber in 1..$outerCount) {
        foreach ($innerNumber in 1..$innerCount) {
            $stages += [pscustomobject]@{
                Key = "outer_${outerNumber}_inner_$innerNumber"
                Outer = "outer_$outerNumber"
                Kind = 'inner'
                Label = if ($Context.SensitivityActive) {
                    "SENS_O${outerNumber}_I$innerNumber"
                } else { "HPO_O${outerNumber}_I$innerNumber" }
            }
        }
        if (-not $Context.SensitivityActive) {
            $stages += [pscustomobject]@{
                Key = "outer_${outerNumber}_evaluation"
                Outer = "outer_$outerNumber"
                Kind = 'outer_evaluation'
                Label = "Eval_O$outerNumber"
            }
        }
    }

    $metadata = @{}
    $terminal = @{}
    foreach ($line in Get-Content -LiteralPath $ledgerPath) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $row = $line | ConvertFrom-Json
        if ($row.status -eq 'proposed') {
            $metadata[[string]$row.trial_id] = $row
        }
        elseif ($row.status -in @('complete', 'failed')) {
            $terminal[[string]$row.trial_id] = $row
        }
    }

    $sensitivityMetadata = @{}
    $sensitivityTerminal = @{}
    $sensitivityLedgerPath = Join-Path $Context.RunDirectory 'sensitivity_trial_ledger.jsonl'
    if (Test-Path -LiteralPath $sensitivityLedgerPath) {
        foreach ($line in Get-Content -LiteralPath $sensitivityLedgerPath) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            $row = $line | ConvertFrom-Json
            if ($row.status -eq 'proposed') {
                $sensitivityMetadata[[string]$row.trial_id] = $row
            }
            elseif ($row.status -in @('complete', 'failed')) {
                $sensitivityTerminal[[string]$row.trial_id] = $row
            }
        }
    }

    $holdoutRows = @()
    $holdoutLedgerPath = Join-Path $Context.RunDirectory 'holdout_fit_ledger.jsonl'
    if (Test-Path -LiteralPath $holdoutLedgerPath) {
        $holdoutRows = @(Get-Content -LiteralPath $holdoutLedgerPath | ForEach-Object {
            if (-not [string]::IsNullOrWhiteSpace($_)) { $_ | ConvertFrom-Json }
        })
    }
    $fitLedgerName = if ($Context.SensitivityActive) {
        'sensitivity_fit_ledger.jsonl'
    } else { 'fit_ledger.jsonl' }
    $fitRows = @()
    $fitLedgerPath = Join-Path $Context.RunDirectory $fitLedgerName
    if (Test-Path -LiteralPath $fitLedgerPath) {
        $fitRows = @(Get-Content -LiteralPath $fitLedgerPath | ForEach-Object {
            if (-not [string]::IsNullOrWhiteSpace($_)) { $_ | ConvertFrom-Json }
        })
    }
    $outerRows = @()
    $outerLedgerPath = Join-Path $Context.RunDirectory 'outer_fit_ledger.jsonl'
    if (Test-Path -LiteralPath $outerLedgerPath) {
        $outerRows = @(Get-Content -LiteralPath $outerLedgerPath | ForEach-Object {
            if (-not [string]::IsNullOrWhiteSpace($_)) { $_ | ConvertFrom-Json }
        })
    }

    $rows = @()
    $estimatedRemaining = 0.0
    $unknownEstimate = $false
    $overallEstimatedRemaining = 0.0
    $overallSensitivityRemaining = 0.0
    $overallHpoRemaining = 0.0
    $overallOuterEvaluationRemaining = 0.0
    $overallFinalEvaluationRemaining = 0.0
    $overallEstimateAvailable = $null -ne $calibration
    foreach ($arm in $arms) {
        $armId = [string]$arm.id
        $scenarioMultiplier = if ($arm.interface -eq 'RLPolicy') {
            $riskScenarioCount
        } else { 1 }
        if ($Context.SensitivityActive -and $null -ne $design) {
            $proposalCount = @($design.arms.$armId.proposals).Count
        }
        elseif ($arm.interface -eq 'RLPolicy') {
            $proposalCount = [int]$protocol.search.budgets.rl_per_family_per_risk_scenario
        }
        elseif ($arm.component_id -match 'lstm|transformer') {
            $proposalCount = [int]$protocol.search.budgets.deep_supervised_per_family
        }
        else {
            $proposalCount = [int]$protocol.search.budgets.classical_per_family
        }
        $armTrials = @($metadata.Values | Where-Object { $_.arm_id -eq $armId })
        $calibrationRecord = if ($null -ne $calibration) {
            $calibration.records | Where-Object { $_.arm_id -eq $armId } |
                Select-Object -First 1
        } else { $null }
        if ($null -eq $calibrationRecord) { $overallEstimateAvailable = $false }
        $armHpoRemainingCells = 0
        $armOuterRemainingCells = 0
        $sensitivityProgress = @{}
        foreach ($fold in $folds) {
            $sensitivityTrials = @($sensitivityMetadata.Values | Where-Object {
                $_.arm_id -eq $armId -and $_.outer_fold_id -eq $fold
            })
            $sensitivityComplete = @($sensitivityTrials | Where-Object {
                $sensitivityTerminal.ContainsKey([string]$_.trial_id) -and
                $sensitivityTerminal[[string]$_.trial_id].status -eq 'complete'
            }).Count
            $sensitivityFailed = @($sensitivityTrials | Where-Object {
                $sensitivityTerminal.ContainsKey([string]$_.trial_id) -and
                $sensitivityTerminal[[string]$_.trial_id].status -eq 'failed'
            }).Count
            $sensitivityExpected = if ($null -ne $design) {
                @($design.arms.$armId.proposals).Count * $scenarioMultiplier
            } else { 0 }
            $sensitivitySeconds = 0.0
            foreach ($trial in $sensitivityTrials) {
                if ($sensitivityTerminal.ContainsKey([string]$trial.trial_id)) {
                    $sensitivitySeconds += [double](
                        $sensitivityTerminal[[string]$trial.trial_id].resource_seconds)
                }
            }
            $sensitivityText = "$sensitivityComplete/$sensitivityExpected"
            if ($sensitivityFailed -gt 0) {
                $sensitivityText += " (!$sensitivityFailed failed)"
            }
            $sensitivityRuntime = Format-Duration $sensitivitySeconds
            if ($sensitivityRuntime -ne '-') {
                $sensitivityText += " ($sensitivityRuntime)"
            }
            $sensitivityProgress[$fold] = $sensitivityText
        }
        $stageProgress = @{}
        foreach ($stage in $stages) {
            $seconds = 0.0
            $observedDurations = @()
            $stageRows = if ($stage.Kind -eq 'inner') {
                @($fitRows | Where-Object {
                    $trialId = [string]$_.trial_id
                    $rowArm = if ($_.arm_id) { [string]$_.arm_id }
                        elseif ($metadata.ContainsKey($trialId)) {
                            [string]$metadata[$trialId].arm_id
                        } else { '' }
                    $rowArm -eq $armId -and $_.outer_fold_id -eq $stage.Outer -and
                        $_.inner_fold_id -eq $stage.Key
                })
            }
            else {
                @($outerRows | Where-Object {
                    $_.arm_id -eq $armId -and $_.outer_fold_id -eq $stage.Outer
                })
            }
            $completeRows = @($stageRows | Where-Object { $_.status -ne 'failed' })
            $failedRows = @($stageRows | Where-Object { $_.status -eq 'failed' })
            foreach ($record in $completeRows) {
                $value = if ($null -ne $record.telemetry.wall_seconds) {
                    [double]$record.telemetry.wall_seconds
                }
                elseif ($null -ne $record.telemetry.duration_seconds) {
                    [double]$record.telemetry.duration_seconds
                } else { 0.0 }
                $seconds += $value
                if ($value -gt 0) { $observedDurations += $value }
            }
            foreach ($record in $failedRows) {
                if ($null -ne $record.resource_seconds) {
                    $seconds += [double]$record.resource_seconds
                }
            }
            $planned = if ($stage.Kind -eq 'inner') {
                $proposalCount * $scenarioMultiplier * $seedCount
            } else { $scenarioMultiplier * $seedCount }
            $completed = $completeRows.Count
            $failed = $failedRows.Count
            $remaining = [math]::Max(0, $planned - $completed - $failed)
            if (-not $Context.SensitivityActive) {
                if ($stage.Kind -eq 'inner') {
                    $armHpoRemainingCells += $remaining
                } else { $armOuterRemainingCells += $remaining }
            }
            $progressText = "$completed/$planned"
            if ($failed -gt 0) { $progressText += " (!$failed failed)" }
            $runtimeText = Format-Duration $seconds
            if ($runtimeText -ne '-') { $progressText += " ($runtimeText)" }
            $stageProgress[$stage.Key] = $progressText
            if ($remaining -gt 0 -and $observedDurations.Count -gt 0) {
                $sorted = @($observedDurations | Sort-Object)
                $middle = [int][math]::Floor($sorted.Count / 2)
                $median = if ($sorted.Count % 2 -eq 0) {
                    ($sorted[$middle - 1] + $sorted[$middle]) / 2
                } else { $sorted[$middle] }
                $estimatedRemaining += $remaining * $median
            }
            elseif ($remaining -gt 0) { $unknownEstimate = $true }
        }

        $finalExpected = $seedCount * $scenarioMultiplier
        $finalComplete = @($holdoutRows | Where-Object { $_.arm_id -eq $armId }).Count
        $armFinalRemainingCells = [math]::Max(0, $finalExpected - $finalComplete)
        if ($Context.SensitivityActive) {
            $armOuterRemainingCells += $outerCount * $finalExpected
        }
        if ($null -ne $calibrationRecord) {
            $observedArmDurations = @($fitRows | Where-Object {
                $trialId = [string]$_.trial_id
                $rowArm = if ($_.arm_id) { [string]$_.arm_id }
                    elseif ($metadata.ContainsKey($trialId)) {
                        [string]$metadata[$trialId].arm_id
                    } else { '' }
                $rowArm -eq $armId -and $_.status -ne 'failed'
            } | ForEach-Object {
                if ($null -ne $_.telemetry.wall_seconds) {
                    [double]$_.telemetry.wall_seconds
                }
                elseif ($null -ne $_.telemetry.duration_seconds) {
                    [double]$_.telemetry.duration_seconds
                }
            } | Where-Object { $_ -gt 0 } | Sort-Object)
            $observedMedian = 0.0
            if ($observedArmDurations.Count -gt 0) {
                $middle = [int][math]::Floor($observedArmDurations.Count / 2)
                $observedMedian = if ($observedArmDurations.Count % 2 -eq 0) {
                    ($observedArmDurations[$middle - 1] + $observedArmDurations[$middle]) / 2
                } else { $observedArmDurations[$middle] }
            }
            $perCellEstimate = [math]::Max(
                [double]$calibrationRecord.wall_seconds, $observedMedian)
            $overallHpoRemaining += $armHpoRemainingCells * $perCellEstimate
            $overallOuterEvaluationRemaining +=
                $armOuterRemainingCells * $perCellEstimate
            $overallFinalEvaluationRemaining +=
                $armFinalRemainingCells * $perCellEstimate
        }
        $finalStarted = $armId -in @($Context.HoldoutStartedArms)
        $finalTraining = if ($finalComplete -ge $finalExpected) {
            "$finalExpected/$finalExpected"
        }
        elseif ($finalStarted) { "$finalComplete/$finalExpected running" }
        else { "$finalComplete/$finalExpected" }
        $holdoutStatus = if ($finalComplete -ge $finalExpected) {
            "$finalExpected/$finalExpected"
        }
        elseif ($finalComplete -gt 0) { "$finalComplete/$finalExpected" }
        elseif ($finalStarted) { "0/$finalExpected waiting" }
        else { "0/$finalExpected" }
        $rows += [pscustomobject]@{
            Arm = $armId
            SensitivityProgress = $sensitivityProgress
            StageProgress = $stageProgress
            FinalTraining = $finalTraining
            Holdout = $holdoutStatus
        }
    }

    $sensitivityCompleteTotal = 0
    $sensitivityExpectedTotal = 0
    foreach ($row in $rows) {
        foreach ($fold in $folds) {
            if ($row.SensitivityProgress[$fold] -match '^(?<complete>\d+)/(?<expected>\d+)') {
                $sensitivityCompleteTotal += [int]$Matches['complete']
                $sensitivityExpectedTotal += [int]$Matches['expected']
            }
        }
    }
    if ($Context.SensitivityActive -and $null -ne $calibration) {
        $sensitivityRemaining = [math]::Max(0,
            $sensitivityExpectedTotal - $sensitivityCompleteTotal)
        $sensitivityFraction = if ($sensitivityExpectedTotal -gt 0) {
            $sensitivityRemaining / $sensitivityExpectedTotal
        } else { 0.0 }
        $tierName = [string]$calibration.budget_tier
        $selectionSeconds = [double]$calibration.estimated_seconds_by_tier.$tierName -
            [double]$calibration.controlled_sensitivity_estimated_seconds
        $overallHpoRemaining += $selectionSeconds
        $overallSensitivityRemaining =
            [double]$calibration.controlled_sensitivity_estimated_seconds *
            $sensitivityFraction
    }
    $overallEstimatedRemaining = $overallSensitivityRemaining +
        $overallHpoRemaining + $overallOuterEvaluationRemaining +
        $overallFinalEvaluationRemaining
    return [pscustomobject]@{
        Rows = $rows
        OuterFolds = $folds
        Stages = $stages
        SensitivityActive = $Context.SensitivityActive
        SensitivityComplete = $sensitivityCompleteTotal
        SensitivityExpected = $sensitivityExpectedTotal
        EstimatedRemaining = $estimatedRemaining
        EstimateIsPartial = $unknownEstimate
        OverallEstimatedRemaining = $overallEstimatedRemaining
        OverallSensitivityRemaining = $overallSensitivityRemaining
        OverallHpoRemaining = $overallHpoRemaining
        OverallOuterEvaluationRemaining = $overallOuterEvaluationRemaining
        OverallFinalEvaluationRemaining = $overallFinalEvaluationRemaining
        OverallEstimateAvailable = $overallEstimateAvailable
        Scope = if ($Context.SensitivityActive) {
            'controlled-sensitivity candidates'
        } else { 'selection-eligible candidates' }
    }
}

function Write-ProgressTable {
    param($Summary)

    if ($null -eq $Summary) {
        Write-Host ' Progression data is not available yet.' -ForegroundColor DarkGray
        return
    }
    $armWidth = [math]::Max(3, ($Summary.Rows | ForEach-Object { $_.Arm.Length } |
        Measure-Object -Maximum).Maximum)
    $stageWidth = [math]::Max(11, ($Summary.Rows | ForEach-Object {
        foreach ($stage in $Summary.Stages) { $_.StageProgress[$stage.Key].Length }
    } | Measure-Object -Maximum).Maximum)
    $finalWidth = [math]::Max(14, ($Summary.Rows | ForEach-Object {
        $_.FinalTraining.Length
    } | Measure-Object -Maximum).Maximum)
    $holdoutWidth = [math]::Max(9, ($Summary.Rows | ForEach-Object {
        $_.Holdout.Length
    } | Measure-Object -Maximum).Maximum)
    $sensitivityWidth = [math]::Max(14, ($Summary.Rows | ForEach-Object {
        foreach ($fold in $Summary.OuterFolds) { $_.SensitivityProgress[$fold].Length }
    } | Measure-Object -Maximum).Maximum)

    Write-Host ("`n Controlled sensitivity (selection-ineligible, {0}/{1} total):" -f
        $Summary.SensitivityComplete, $Summary.SensitivityExpected) -ForegroundColor Magenta
    Write-Host ' Objective: measure marginal parameter effects; excluded from finalist selection.' -ForegroundColor DarkGray
    $sensitivityHeader = ' {0,-' + $armWidth + '}'
    $sensitivityValues = @('Arm')
    foreach ($fold in $Summary.OuterFolds) {
        $sensitivityHeader += '  {' + $sensitivityValues.Count + ',-' +
            $sensitivityWidth + '}'
        $outerNumber = $fold -replace 'outer_', ''
        $sensitivityValues += "Outer $outerNumber / Inner 1"
    }
    Write-Host ($sensitivityHeader -f $sensitivityValues) -ForegroundColor Cyan
    Write-Host (' ' + ('-' * $armWidth) +
        (($Summary.OuterFolds | ForEach-Object {
            '  ' + ('-' * $sensitivityWidth)
        }) -join '')) -ForegroundColor DarkGray
    foreach ($row in $Summary.Rows) {
        $format = ' {0,-' + $armWidth + '}'
        $values = @($row.Arm)
        foreach ($fold in $Summary.OuterFolds) {
            $format += '  {' + $values.Count + ',-' + $sensitivityWidth + '}'
            $values += $row.SensitivityProgress[$fold]
        }
        Write-Host ($format -f $values)
    }

    if ($Summary.SensitivityActive) {
        return
    }

    $outerGroups = @($Summary.Stages | Group-Object Outer)
    foreach ($group in $outerGroups) {
        $outerLabel = $group.Name -replace '_', ' '
        Write-Host ("`n {0}:" -f ([Globalization.CultureInfo]::CurrentCulture.TextInfo.ToTitleCase(
            $outerLabel))) -ForegroundColor Magenta
        Write-Host ' Objective: tune on inner folds; assess the selection on the unseen outer window.' -ForegroundColor DarkGray
        $header = ' {0,-' + $armWidth + '}'
        $arguments = @('Arm')
        foreach ($stage in $group.Group) {
            $header += '  {' + $arguments.Count + ',-' + $stageWidth + '}'
            $arguments += if ($stage.Kind -eq 'inner') {
                'Inner ' + ($stage.Key -replace '.*_inner_', '')
            } else { 'Outer eval' }
        }
        Write-Host ($header -f $arguments) -ForegroundColor Cyan
        $underline = ' ' + ('-' * $armWidth) +
            (($group.Group | ForEach-Object { '  ' + ('-' * $stageWidth) }) -join '')
        Write-Host $underline -ForegroundColor DarkGray
        foreach ($row in $Summary.Rows) {
            $format = ' {0,-' + $armWidth + '}'
            $values = @($row.Arm)
            foreach ($stage in $group.Group) {
                $format += '  {' + $values.Count + ',-' + $stageWidth + '}'
                $values += $row.StageProgress[$stage.Key]
            }
            Write-Host ($format -f $values)
        }
    }

    Write-Host "`n Final evaluation:" -ForegroundColor Magenta
    Write-Host ' Objective: refit locked models; report holdout without further selection.' -ForegroundColor DarkGray
    $finalHeader = ' {0,-' + $armWidth + '}  {1,-' + $finalWidth +
        '}  {2,-' + $holdoutWidth + '}'
    Write-Host ($finalHeader -f 'Arm', 'Final training', 'Holdout') -ForegroundColor Cyan
    Write-Host (' ' + ('-' * $armWidth) + '  ' + ('-' * $finalWidth) +
        '  ' + ('-' * $holdoutWidth)) -ForegroundColor DarkGray
    foreach ($row in $Summary.Rows) {
        Write-Host ($finalHeader -f $row.Arm, $row.FinalTraining, $row.Holdout)
    }
}

function Write-ObservedEta {
    param(
        $Summary,
        [pscustomobject]$Context
    )

    if ($Context.Phase -eq 'COMPLETED') {
        Write-Host ' Complete (the study has finished).' -ForegroundColor Green
        return
    }
    if ($Context.Phase -eq 'FAILED') {
        Write-Host ' Unavailable (the study stopped with an error).' -ForegroundColor Red
        return
    }
    if ($null -eq $Summary) {
        Write-Host ' Unavailable: run progression data has not been written yet.' -ForegroundColor DarkGray
        return
    }
    if (-not $Summary.OverallEstimateAvailable) {
        Write-Host ' Unavailable: verified score-blind calibration evidence was not found.' -ForegroundColor DarkGray
        return
    }
    $sensitivitySeconds = Get-RoundedEstimateSeconds $Summary.OverallSensitivityRemaining
    $hpoSeconds = Get-RoundedEstimateSeconds $Summary.OverallHpoRemaining
    $outerSeconds = Get-RoundedEstimateSeconds $Summary.OverallOuterEvaluationRemaining
    $finalSeconds = Get-RoundedEstimateSeconds $Summary.OverallFinalEvaluationRemaining
    $eta = Format-RoundedEstimate (
        $sensitivitySeconds + $hpoSeconds + $outerSeconds + $finalSeconds)
    Write-Host (' About {0}' -f $eta) -ForegroundColor Yellow
    if ($Summary.OverallSensitivityRemaining -gt 0) {
        Write-Host ('   Controlled sensitivity : about {0}' -f
            (Format-RoundedEstimate $sensitivitySeconds))
    }
    Write-Host ('   Inner HPO cells         : about {0}' -f
        (Format-RoundedEstimate $hpoSeconds))
    Write-Host ('   Outer evaluations       : about {0}' -f
        (Format-RoundedEstimate $outerSeconds))
    Write-Host ('   Final fit and holdout    : about {0}' -f
        (Format-RoundedEstimate $finalSeconds))
    Write-Host ' Basis: future cells use observed per-arm medians with verified' -ForegroundColor DarkGray
    Write-Host ' score-blind production calibration as a conservative fallback.' -ForegroundColor DarkGray
}

function Write-EventFields {
    param([string]$Line)

    if ([string]::IsNullOrWhiteSpace($Line)) {
        Write-Host ' Event            : Awaiting first event'
        return
    }
    $match = [regex]::Match($Line,
        '^(?<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(?<level>\S+)\s+(?<message>.*)$')
    if (-not $match.Success) {
        Write-Host (' Event            : {0}' -f $Line)
        return
    }
    $message = $match.Groups['message'].Value
    $keyMatch = [regex]::Match($message,
        '\s(?:run_id|candidate|cells|arm|fold|scenario|trial|seed|elapsed_seconds|error)=')
    $eventName = if ($keyMatch.Success) {
        $message.Substring(0, $keyMatch.Index).Trim()
    } else { $message }
    $fields = [ordered]@{
        Timestamp = $match.Groups['timestamp'].Value
        Level = $match.Groups['level'].Value
        Event = $eventName
    }
    foreach ($key in @('run_id', 'candidate', 'cells', 'arm', 'fold', 'scenario',
        'trial', 'seed', 'elapsed_seconds', 'error')) {
        $valueMatch = [regex]::Match($message, "(?:^|\s)$key=(?<value>\S+)")
        if ($valueMatch.Success) { $fields[$key] = $valueMatch.Groups['value'].Value }
    }
    foreach ($field in $fields.GetEnumerator()) {
        Write-Host (' {0,-16} : {1}' -f $field.Key, $field.Value)
    }
}

function Write-RunStatus {
    param(
        [pscustomobject]$Context,
        [string]$EventLine,
        [double]$RunSeconds
    )

    Write-Host (' Phase            : {0}' -f $Context.Phase)
    Write-Host (' Activity         : {0}' -f $Context.Activity)
    if (-not [string]::IsNullOrWhiteSpace($Context.OperationalWarning)) {
        Write-Host (' Warning          : {0}' -f $Context.OperationalWarning) -ForegroundColor Yellow
    }
    Write-Host (' Run duration     : {0}' -f (Format-Duration $RunSeconds))
    Write-EventFields -Line $EventLine
}

do {
    try {
        $metricLine = Get-Content -LiteralPath $metricsPath -Tail 1 -ErrorAction Stop
        if ([string]::IsNullOrWhiteSpace($metricLine)) {
            throw 'Waiting for the first resource heartbeat'
        }
        $metrics = $metricLine | ConvertFrom-Json
        $eventLine = Get-LatestEventLine -Path $stderrPath
        $context = Get-RunContext -Path $stderrPath -EventsPath $eventsPath
        $armProgress = Get-ArmProgress -Context $context
        $gpu = $metrics.gpu.devices | Select-Object -First 1
        $normalisedCpu = $metrics.process_tree.cpu_utilization_percent /
            [Environment]::ProcessorCount

        Clear-MonitorScreen
        Write-Host $banner -ForegroundColor Cyan
        Write-Host (' Log: {0}' -f $resolvedLogDir) -ForegroundColor DarkGray
        Write-Host ('-' * 78) -ForegroundColor DarkCyan

        Write-Host ' Run status:' -ForegroundColor Yellow
        Write-RunStatus -Context $context -EventLine $eventLine `
            -RunSeconds ([double]$metrics.elapsed_seconds)

        Write-Host ('-' * 78) -ForegroundColor DarkCyan
        Write-Host ' Overall estimated time remaining:' -ForegroundColor Yellow
        Write-ObservedEta -Summary $armProgress -Context $context

        Write-Host ('-' * 78) -ForegroundColor DarkCyan
        Write-Host ' Progress by model arm:' -ForegroundColor Yellow
        Write-ProgressTable -Summary $armProgress
        if ($null -ne $armProgress) {
            Write-Host (' Scope: {0}; fold columns are cumulative candidate runtime.' -f
                $armProgress.Scope) -ForegroundColor DarkGray
        }

        Write-Host ('-' * 78) -ForegroundColor DarkCyan
        Write-Host ' System status:' -ForegroundColor Yellow
        [pscustomobject]@{
            ProcessCPU = '{0:N1}%' -f $normalisedCpu
            SystemCPU = '{0:N1}%' -f $metrics.system.cpu_utilization_percent
            RAM = '{0:N2} / {1:N2} GiB (process / available)' -f
                ($metrics.process_tree.rss_bytes / 1GB),
                ($metrics.system.ram_available_bytes / 1GB)
            GPU = if ($null -ne $gpu) { '{0:N1}%' -f $gpu.utilization_percent } else { 'Unavailable' }
            VRAM = if ($null -ne $gpu) {
                '{0:N0} / {1:N0} MiB' -f $gpu.vram_used_mib, $gpu.vram_total_mib
            } else { 'Unavailable' }
        } | Format-List
        Write-Host "`n Refresh: ${RefreshSeconds}s | Ctrl+C closes this monitor only" -ForegroundColor DarkGray
    }
    catch {
        Clear-MonitorScreen
        Write-Host $banner -ForegroundColor Cyan
        Write-Host " Waiting for monitoring data: $($_.Exception.Message)" -ForegroundColor Yellow
    }

    if (-not $Once) {
        Start-Sleep -Seconds $RefreshSeconds
    }
} while (-not $Once)
