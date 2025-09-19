# PowerShell script to guide the installation of liblouis on Windows

Write-Host "=== Liblouis Installation Guide for Windows ===" -ForegroundColor Cyan
Write-Host "Ensure you have:" 
Write-Host "1. Git for Windows: https://git-scm.com/download/win"
Write-Host "2. Visual Studio with C++ Build Tools: https://visualstudio.microsoft.com/downloads/"
Write-Host "3. Python for Windows: https://www.python.org/downloads/windows/"
Write-Host "----------------------------------------------"
Read-Host -Prompt "Press Enter to continue"

# 1. Clone Liblouis
if (-not (Test-Path -Path "liblouis")) {
    git clone https://github.com/liblouis/liblouis.git
} else {
    Write-Host "Liblouis repository already exists."
}

# 2. Navigate to windows directory
Set-Location "liblouis/windows"

# 3. Guide to build DLL
Write-Host "----------------------------------------------"
Write-Host "To build liblouis.dll:"
Write-Host "1. Open 'Developer Command Prompt for VS'."
Write-Host "2. Navigate here: cd '$((Get-Location).Path)'"
Write-Host "3. Run: nmake /f Makefile.nmake"
Write-Host "----------------------------------------------"
Read-Host -Prompt "Press Enter after DLL is built"

# 4. Install Python bindings
Set-Location "../python"
python -m pip install .
Write-Host "Python module installation complete."

# 5. Add DLL to PATH
$dllPath = (Resolve-Path "../windows").Path
Write-Host "Adding DLL path to system PATH..."
[Environment]::SetEnvironmentVariable(
    "Path", 
    $env:Path + ";$dllPath", 
    [EnvironmentVariableTarget]::Machine
)
Write-Host "DLL path added. You may need to restart your terminal."
