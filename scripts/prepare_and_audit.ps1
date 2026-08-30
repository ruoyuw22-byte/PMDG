[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $AdditionalArguments
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $ProjectRoot
try {
    & python -m dagl_mil.cli prepare @AdditionalArguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & python -m dagl_mil.cli audit
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
