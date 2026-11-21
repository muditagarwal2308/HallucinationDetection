<#
.SYNOPSIS
  Exports the milestone decks to PDF (and optionally every slide to PNG) with Microsoft PowerPoint.

.EXAMPLE
  node presentations/build_decks.js
  powershell -NoProfile -ExecutionPolicy Bypass -File presentations/export_pdf.ps1

.EXAMPLE
  # also render slide images for visual QA into a folder outside the repository
  powershell -File presentations/export_pdf.ps1 -PngDir C:\temp\qa_decks -Only M2,M4
#>
param(
  [string]$PngDir = "",
  [string[]]$Only = @(),
  [int]$PngWidth = 1600,
  [int]$PngHeight = 900
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$pdfDir = Join-Path $here "pdf"
New-Item -ItemType Directory -Force -Path $pdfDir | Out-Null

$decks = Get-ChildItem -Path $here -Filter "M?_*.pptx" | Sort-Object Name
if ($Only.Count -gt 0) {
  $decks = $decks | Where-Object { $prefix = $_.Name.Substring(0, 2); $Only -contains $prefix }
}
if (-not $decks) { Write-Error "No decks found - run 'node presentations/build_decks.js' first."; exit 1 }

$pp = New-Object -ComObject PowerPoint.Application
try {
  foreach ($deck in $decks) {
    # ReadOnly, Untitled, WithWindow = false
    $pres = $pp.Presentations.Open($deck.FullName, $true, $false, $false)
    try {
      $pdf = Join-Path $pdfDir ($deck.BaseName + ".pdf")
      $pres.SaveAs($pdf, 32)   # ppSaveAsPDF
      Write-Output ("{0}: {1} slides -> {2}" -f $deck.Name, $pres.Slides.Count, (Resolve-Path -Relative $pdf))
      if ($PngDir) {
        $out = Join-Path $PngDir $deck.BaseName
        New-Item -ItemType Directory -Force -Path $out | Out-Null
        for ($i = 1; $i -le $pres.Slides.Count; $i++) {
          $png = Join-Path $out ("slide-{0:D2}.png" -f $i)
          $pres.Slides.Item($i).Export($png, "PNG", $PngWidth, $PngHeight)
        }
      }
    }
    finally {
      $pres.Close()
      [System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres) | Out-Null
    }
  }
}
finally {
  # Quit only if no other presentation is open (do not close the user's own work).
  if ($pp.Presentations.Count -eq 0) { $pp.Quit() }
  [System.Runtime.InteropServices.Marshal]::ReleaseComObject($pp) | Out-Null
  [GC]::Collect()
  [GC]::WaitForPendingFinalizers()
}
