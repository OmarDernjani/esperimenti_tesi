param(
    [ValidateSet('plan', 'run')][string]$Action = 'plan',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ClientArgs
)
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '../../..')
try {
    $portable = Join-Path (Get-Location) 'CodePromptOpt/.runtime/python/python.exe'
    $pythonCommand = if (Test-Path $portable) { $portable } else { 'python' }
    $moduleArgs = @('-m', 'CodePromptOpt.finetuning.downstream.run')
    if ($Action -eq 'plan') { $moduleArgs += '--plan-only' }
    & $pythonCommand @moduleArgs @ClientArgs
    $clientExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $clientExit
