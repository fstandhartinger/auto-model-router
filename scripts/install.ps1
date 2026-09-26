<#
.SYNOPSIS
  One-command installer for the auto model router on Windows 11 (PowerShell 5.1 or 7).

.DESCRIPTION
  What it does, and nothing else:
    * fetches github.com/fstandhartinger/auto-model-router (MIT) at -Ref (default main)
      into %USERPROFILE%\.auto-router\src (git if present, else the GitHub zip)
    * makes a virtualenv in %USERPROFILE%\.auto-router\venv and installs the router's
      dependencies into it (no administrator rights, nothing system-wide)
    * writes %USERPROFILE%\.auto-router\config.yaml and launcher.yaml (keys by
      environment variable NAME only)
    * writes one launcher, %USERPROFILE%\.auto-router\bin\auto-router.cmd, and adds that
      folder to your user PATH only if you agree (or pass -Yes); -Uninstall removes it
    * configures the harnesses you choose; every edited file is backed up next to itself
      (*.auto-router-bak-*) and recorded, and -Uninstall restores it
    * downloads a local model only with -WithBonsai / -WithJevLocal, after showing its
      size and getting a yes (or -Yes)

  It never reads, prints or writes an API key value. Read it before you run it.

  On native Windows the HTTP router, the doctor and the harness provider entries work.
  The MCP delegate tool and route-run need POSIX process control: use WSL for those
  (run scripts/install.sh inside WSL).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File install.ps1 -Yes -Models cloud -Harness opencode,copilot
.EXAMPLE
  & ([scriptblock]::Create((irm https://raw.githubusercontent.com/fstandhartinger/auto-model-router/main/scripts/install.ps1))) -DryRun -Models all -Harness all
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File install.ps1 -Uninstall
#>
[CmdletBinding()]
param(
  [switch]$Yes,
  [string]$Models,
  [string]$Harness,
  [ValidateSet('auto', 'jev-local', 'hosted', 'heuristic', 'laya')][string]$Classifier = 'auto',
  [switch]$WithBonsai,
  [switch]$WithJevLocal,
  [switch]$ClaudeGateway,
  [string]$Project,
  [switch]$NoDelegate,
  [string]$Ref = $(if ($env:AUTO_ROUTER_REF) { $env:AUTO_ROUTER_REF } else { 'main' }),
  [int]$Port = $(if ($env:AUTO_ROUTER_PORT) { [int]$env:AUTO_ROUTER_PORT } else { 8787 }),
  [switch]$Force,
  [switch]$DryRun,
  [switch]$Uninstall,
  [switch]$Purge,
  [switch]$Update,
  [switch]$NoPath,
  [switch]$Json
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$Repo = if ($env:AUTO_ROUTER_REPO) { $env:AUTO_ROUTER_REPO } else { 'https://github.com/fstandhartinger/auto-model-router.git' }
$UserHome = if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME }
$Root = if ($env:AUTO_ROUTER_HOME) { $env:AUTO_ROUTER_HOME } else { Join-Path $UserHome '.auto-router' }
$Src = Join-Path $Root 'src'
$Venv = Join-Path $Root 'venv'
$BinDir = Join-Path $Root 'bin'
$OnWindows = ($PSVersionTable.PSEdition -eq 'Desktop') -or ((Test-Path variable:IsWindows) -and $IsWindows)
$Launcher = if ($OnWindows) { Join-Path $BinDir 'auto-router.cmd' } else { Join-Path $BinDir 'auto-router' }
$VenvPy = if ($OnWindows) { Join-Path $Venv 'Scripts\python.exe' } else { Join-Path $Venv 'bin/python' }
$Mark = 'auto-router launcher (written by install.ps1)'
$PathRecord = Join-Path $Root 'path-added.txt'

function Say([string]$Text) { if (-not $Json) { Write-Host "  $Text" } }
function Fail([string]$Text) { Write-Host "`n  install.ps1: $Text`n" -ForegroundColor Red; exit 1 }

if ($Ref -notmatch '^[A-Za-z0-9._/-]+$' -or $Ref.StartsWith('-')) { Fail '-Ref may only contain letters, digits and . _ / -' }
if (-not $Json) { Write-Host "`n  auto-model-router installer (Windows PowerShell $($PSVersionTable.PSVersion))`n  ------------------------------" }
if ($DryRun) { Say 'DRY RUN: nothing will be written, downloaded or edited.' }

function Find-Python {
  $candidates = @()
  if (Get-Command py -ErrorAction SilentlyContinue) { $candidates += , @('py', '-3') }
  foreach ($n in 'python3', 'python') {
    $c = Get-Command $n -ErrorAction SilentlyContinue
    # The Microsoft Store stub under WindowsApps opens the Store instead of running.
    if ($c -and $c.Source -notmatch 'WindowsApps') { $candidates += , @($c.Source) }
  }
  foreach ($cand in $candidates) {
    $exe = $cand[0]; $pre = @($cand | Select-Object -Skip 1)
    try {
      & $exe @pre -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>$null
      if ($LASTEXITCODE -eq 0) { return , $cand }
    } catch { continue }
  }
  return $null
}

function Invoke-Installer([string[]]$PyArgs, [string]$Python, [string[]]$Pre, [string]$CodeDir) {
  $old = $env:PYTHONPATH
  $env:PYTHONPATH = $CodeDir
  $env:AUTO_ROUTER_HOME = $Root
  try { & $Python @Pre -m auto_router.installer @PyArgs; return $LASTEXITCODE }
  finally { $env:PYTHONPATH = $old }
}

function Get-Code([string]$Target) {
  if (Get-Command git -ErrorAction SilentlyContinue) {
    if (-not (Test-Path (Join-Path $Target '.git'))) {
      if (Test-Path $Target) { Remove-Item -Recurse -Force $Target }
      git init --quiet $Target | Out-Null
      git -C $Target remote add origin $Repo
    }
    git -C $Target fetch --quiet --depth 1 origin $Ref
    if ($LASTEXITCODE -ne 0) { Fail "git could not fetch $Ref from $Repo" }
    git -C $Target -c advice.detachedHead=false checkout --quiet FETCH_HEAD
    Say "code: $Repo @ $Ref ($(git -C $Target rev-parse --short HEAD))"
    return
  }
  $base = ($Repo -replace '\.git$', '') -replace '^https://github.com/', 'https://codeload.github.com/'
  $zip = "$Target.download.zip"
  Invoke-WebRequest -UseBasicParsing -Uri "$base/zip/$Ref" -OutFile $zip
  $tmp = "$Target.extract"
  if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
  Expand-Archive -Path $zip -DestinationPath $tmp
  if (Test-Path $Target) { Remove-Item -Recurse -Force $Target }
  Move-Item (Get-ChildItem $tmp | Select-Object -First 1).FullName $Target
  Remove-Item -Recurse -Force $tmp, $zip
  Say "code: $base/zip/$Ref (no git; zip)"
}

function Get-UserPath { [Environment]::GetEnvironmentVariable('Path', 'User') }

# ------------------------------------------------------------------ uninstall
if ($Uninstall) {
  if ((Test-Path $VenvPy) -and (Test-Path (Join-Path $Src 'auto_router'))) {
    $a = @('--uninstall'); if ($DryRun) { $a += '--dry-run' }
    [void](Invoke-Installer $a $VenvPy @() $Src)
  } else { Say "no virtualenv at ${Venv}: harness edits (if any) are listed in $Root\install-manifest.json" }
  if ((Test-Path $Launcher) -and (Select-String -Quiet -SimpleMatch $Mark -Path $Launcher)) {
    if ($DryRun) { Say "[dry-run] would remove $Launcher" } else { Remove-Item -Force $Launcher; Say "removed $Launcher" }
  }
  if ($OnWindows -and (Test-Path $PathRecord)) {
    $added = (Get-Content $PathRecord -Raw).Trim()
    $parts = @((Get-UserPath) -split ';' | Where-Object { $_ -and $_ -ne $added })
    if ($DryRun) { Say "[dry-run] would remove $added from your user PATH" }
    else { [Environment]::SetEnvironmentVariable('Path', ($parts -join ';'), 'User'); Remove-Item $PathRecord; Say "removed $added from your user PATH" }
  }
  foreach ($d in $Venv, $Src) {
    if (Test-Path $d) { if ($DryRun) { Say "[dry-run] would remove $d" } else { Remove-Item -Recurse -Force $d; Say "removed $d" } }
  }
  if ($Purge -and (Test-Path $Root)) {
    if ($DryRun) { Say "[dry-run] would remove $Root" } else { Remove-Item -Recurse -Force $Root; Say "removed $Root" }
  } elseif (Test-Path $Root) { Say "kept $Root (config, downloaded models, manifest); -Purge removes it" }
  exit 0
}

# -------------------------------------------------------------- requirements
$py = Find-Python
if (-not $py) { Fail 'Python 3.10 or newer is needed. Install it from python.org or with: winget install Python.Python.3.12' }
$PyExe = $py[0]; $PyPre = @($py | Select-Object -Skip 1)
Say "python: $(& $PyExe @PyPre --version 2>&1)"

$pyArgs = @('--classifier', $Classifier, '--router-url', "http://127.0.0.1:$Port", '--launcher', $Launcher, '--src', $Src)
if ($Yes) { $pyArgs += '--yes' }
if ($DryRun) { $pyArgs += '--dry-run' }
if ($Force) { $pyArgs += '--force' }
if ($Json) { $pyArgs += '--json' }
if ($Models) { $pyArgs += @('--models', $Models) }
if ($Harness) { $pyArgs += @('--harness', $Harness) }
if ($Project) { $pyArgs += @('--project', $Project) }
if ($WithBonsai) { $pyArgs += '--with-bonsai' }
if ($WithJevLocal) { $pyArgs += '--with-jev-local' }
if ($ClaudeGateway) { $pyArgs += '--claude-gateway' }
if ($NoDelegate) { $pyArgs += '--no-delegate' }

# ------------------------------------------------------------------- dry run
if ($DryRun) {
  $code = $Src
  $tmp = $null
  if (-not (Test-Path (Join-Path $Src 'auto_router'))) {
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("auto-router-dry-" + [guid]::NewGuid().ToString('N'))
    Get-Code $tmp
    $code = $tmp
  }
  Say "[dry-run] would install into $Root (code, venv, config) and write $Launcher"
  $runPy = $PyExe; $runPre = $PyPre
  if (Test-Path $VenvPy) { $runPy = $VenvPy; $runPre = @() }
  try { $rc = Invoke-Installer $pyArgs $runPy $runPre $code } finally { if ($tmp) { Remove-Item -Recurse -Force $tmp } }
  exit $rc
}

# ------------------------------------------------------------------ the code
New-Item -ItemType Directory -Force -Path $Root, $BinDir | Out-Null
Get-Code $Src

# ------------------------------------------------------------- the virtualenv
if (-not (Test-Path $VenvPy)) {
  Say "creating a virtualenv in $Venv"
  & $PyExe @PyPre -m venv $Venv
  if ($LASTEXITCODE -ne 0) { Fail 'could not create a virtualenv' }
}
Say 'installing dependencies (this can take a minute)'
& $VenvPy -m pip install --quiet --disable-pip-version-check -r (Join-Path $Src 'requirements.txt')
if ($LASTEXITCODE -ne 0) { Fail "installing the router's dependencies failed" }
if ($Classifier -eq 'laya') {
  Say 'installing the Laya classifier (CPU torch; about 1.7 GB of weights on first use)'
  & $VenvPy -m pip install --quiet 'torch>=2.0' 'laya>=0.3.4'
}

# ----------------------------------------------------------------- launcher
if ((Test-Path $Launcher) -and -not (Select-String -Quiet -SimpleMatch $Mark -Path $Launcher)) {
  if (-not $Force) { Fail "$Launcher exists and was not written by this installer; move it away or use -Force" }
  Copy-Item $Launcher "$Launcher.auto-router-bak-$(Get-Date -Format yyyyMMddTHHmmss)"
}
if ($OnWindows) {
  $cmd = @"
@echo off
rem $Mark
rem   auto-router                 start the router on http://127.0.0.1:$Port/v1
rem   auto-router doctor [--live] check config, router, classifier, routes and harness files
rem   auto-router claude [args]   Claude Code through the router with its own gateway credential
rem   auto-router copilot [args]  GitHub Copilot CLI with the router as its BYOK provider
rem   auto-router hardware        what this machine can run locally
rem   auto-router uninstall       undo harness edits and remove the install
rem   (route-run, the delegate tool and switch mode: use WSL)
setlocal
set "PYTHONPATH=$Src"
set "AUTO_ROUTER_HOME=$Root"
if not defined AUTO_ROUTER_CACHE_DIR set "AUTO_ROUTER_CACHE_DIR=$Root\cache"
if not defined AUTO_ROUTER_PORT set "AUTO_ROUTER_PORT=$Port"
set "URL=http://127.0.0.1:%AUTO_ROUTER_PORT%"
set "CMD=%~1"
if "%CMD%"=="" set "CMD=serve"
if not "%~1"=="" shift
if /I "%CMD%"=="serve" (
  if not defined AUTO_ROUTER_CONFIG set "AUTO_ROUTER_CONFIG=$Root\config.yaml"
  cd /d "$Src"
  "$VenvPy" -m uvicorn auto_router.server:app --host 127.0.0.1 --port %AUTO_ROUTER_PORT%
  exit /b %ERRORLEVEL%
)
if /I "%CMD%"=="doctor" (
  if not defined AUTO_ROUTER_CONFIG set "AUTO_ROUTER_CONFIG=$Root\config.yaml"
  set "AUTO_ROUTER_URL=%URL%"
  "$VenvPy" -m auto_router.smoke %1 %2 %3 %4
  exit /b %ERRORLEVEL%
)
if /I "%CMD%"=="claude" (
  set "ANTHROPIC_BASE_URL=%URL%"
  if not defined AUTO_ROUTER_CLAUDE_KEY set "AUTO_ROUTER_CLAUDE_KEY=local-router"
  set "ANTHROPIC_API_KEY=%AUTO_ROUTER_CLAUDE_KEY%"
  claude %1 %2 %3 %4 %5 %6 %7 %8 %9
  exit /b %ERRORLEVEL%
)
if /I "%CMD%"=="copilot" (
  set "COPILOT_PROVIDER_BASE_URL=%URL%/v1"
  set "COPILOT_PROVIDER_TYPE=openai"
  if not defined COPILOT_MODEL set "COPILOT_MODEL=auto"
  copilot %1 %2 %3 %4 %5 %6 %7 %8 %9
  exit /b %ERRORLEVEL%
)
if /I "%CMD%"=="hardware" ( "$VenvPy" -m auto_router.hardware %1 & exit /b %ERRORLEVEL% )
if /I "%CMD%"=="harness" ( "$VenvPy" -m auto_router.harness %1 %2 %3 %4 & exit /b %ERRORLEVEL% )
if /I "%CMD%"=="uninstall" ( powershell -NoProfile -ExecutionPolicy Bypass -File "$Src\scripts\install.ps1" -Uninstall & exit /b %ERRORLEVEL% )
if /I "%CMD%"=="update" ( powershell -NoProfile -ExecutionPolicy Bypass -File "$Src\scripts\install.ps1" -Update & exit /b %ERRORLEVEL% )
echo auto-router: %CMD% is not available on native Windows (route-run, delegate, switch: use WSL) 1>&2
exit /b 2
"@
  Set-Content -Path $Launcher -Value $cmd -Encoding ASCII
} else {
  # PowerShell on Linux/macOS (used for testing this script): a small sh launcher.
  $sh = @"
#!/bin/sh
# $Mark
export PYTHONPATH="$Src" AUTO_ROUTER_HOME="$Root"
case "`${1:-serve}" in
  serve) export AUTO_ROUTER_CONFIG="`${AUTO_ROUTER_CONFIG:-$Root/config.yaml}"; cd "$Src"; exec "$VenvPy" -m uvicorn auto_router.server:app --host 127.0.0.1 --port $Port ;;
  doctor) shift; export AUTO_ROUTER_CONFIG="`${AUTO_ROUTER_CONFIG:-$Root/config.yaml}"; exec "$VenvPy" -m auto_router.smoke "`$@" ;;
  hardware) shift; exec "$VenvPy" -m auto_router.hardware "`$@" ;;
  *) echo "use scripts/install.sh on this OS" >&2; exit 2 ;;
esac
"@
  Set-Content -Path $Launcher -Value $sh -Encoding ASCII
  if (Get-Command chmod -ErrorAction SilentlyContinue) { chmod +x $Launcher }
}
Say "launcher: $Launcher"

if ($OnWindows -and -not $NoPath) {
  $userPath = Get-UserPath
  if (-not (($userPath -split ';') -contains $BinDir)) {
    $ok = $Yes
    if (-not $ok -and [Environment]::UserInteractive -and -not $Json) {
      $ok = (Read-Host "  Add $BinDir to your user PATH? [y/N]") -match '^[yY]'
    }
    if ($ok) {
      [Environment]::SetEnvironmentVariable('Path', (($userPath, $BinDir) -join ';').Trim(';'), 'User')
      Set-Content -Path $PathRecord -Value $BinDir
      Say "added $BinDir to your user PATH (new terminals see it; -Uninstall removes it)"
    } else { Say "not on PATH: call $Launcher directly" }
  }
}

if ($Update) { Say 'updated; configuration left as it was'; exit 0 }

# ----------------------------------------------------- configure (Python half)
$rc = Invoke-Installer $pyArgs $VenvPy @() $Src
if (-not $Json) {
  Write-Host "`n  Installed.`n"
  Say "start the router:  auto-router          (http://127.0.0.1:$Port/v1)"
  Say 'check everything:  auto-router doctor   (add --live to send one tiny request per route)'
  Say 'undo everything:   auto-router uninstall'
  Say 'route-run, the MCP delegate tool and switch mode: install inside WSL with scripts/install.sh'
  Write-Host ''
}
exit $rc
