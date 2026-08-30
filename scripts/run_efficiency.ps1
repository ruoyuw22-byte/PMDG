[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $AdditionalArguments
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $ProjectRoot
try {
    & python -m dagl_mil.cli efficiency --skip-completed @AdditionalArguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
