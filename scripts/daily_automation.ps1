<# 
.SYNOPSIS
    Runs the PurffleShorts daily automation via Windows Task Scheduler.

.DESCRIPTION
    This script activates the Python virtual environment and runs the daily automation command.
    It includes proper error handling, logging, and prevents overlapping runs.

.NOTES
    Author: PurffleShorts
    Version: 1.0
#>

param(
    [string]$ProjectPath = "C:\Users\Robinson\Documents\purffle-shorts",
    [string]$VenvPath = "C:\Users\Robinson\Documents\purffle-shorts\venv",
    [switch]$DryRun,
    [switch]$SkipQuality,
    [switch]$SkipUpload,
    [switch]$Instagram,
    [string]$EnvFile = ""
)

# Set strict error handling
$ErrorActionPreference = "Stop"

# Function to write timestamped log messages
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Write-Host $logMessage
    
    # Also write to log file
    $logFile = Join-Path $ProjectPath "data\logs\daily_automation.log"
    $logDir = Split-Path $logFile
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
    Add-Content -Path $logFile -Value $logMessage
}

# Function to check if another instance is running
function Check-OverlappingInstance {
    $processName = "purffle-shorts daily"
    $processes = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { 
        $_.CommandLine -like "*purffle_shorts*daily*" 
    }
    return $processes.Count -gt 1
}

try {
    Write-Log "Starting PurffleShorts daily automation"
    
    # Change to project directory
    Set-Location -Path $ProjectPath
    Write-Log "Working directory: $ProjectPath"
    
    # Check for overlapping instances
    if (Check-OverlappingInstance) {
        Write-Log "Another daily automation instance is already running. Exiting." "WARNING"
        exit 1
    }
    
    # Activate virtual environment
    $activateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        Write-Log "Activating virtual environment: $VenvPath"
        & $activateScript
    } else {
        Write-Log "Virtual environment not found at $VenvPath" "ERROR"
        exit 1
    }
    
    # Build command arguments
    $args = @("daily")
    if ($DryRun) { $args += "--dry-run" }
    if ($SkipQuality) { $args += "--skip-quality" }
    if ($SkipUpload) { $args += "--skip-upload" }
    if ($Instagram) { $args += "--instagram" }
    if ($EnvFile) { $args += "--env-file"; $args += $EnvFile }
    
    Write-Log "Running: python -m purffle_shorts $($args -join ' ')"
    
    # Run the daily automation
    $exitCode = 0
    try {
        python -m purffle_shorts @args
        $exitCode = $LASTEXITCODE
    } catch {
        $exitCode = 1
        Write-Log "Exception occurred: $_" "ERROR"
    }
    
    # Interpret exit codes
    switch ($exitCode) {
        0 { Write-Log "Daily automation completed successfully" }
        1 { Write-Log "Daily automation failed" "ERROR" }
        2 { Write-Log "Configuration error" "ERROR" }
        130 { Write-Log "Stopped by user (Ctrl+C)" "WARNING" }
        default { Write-Log "Unknown exit code: $exitCode" "WARNING" }
    }
    
    exit $exitCode
    
} catch {
    Write-Log "Fatal error: $_" "ERROR"
    exit 1
}