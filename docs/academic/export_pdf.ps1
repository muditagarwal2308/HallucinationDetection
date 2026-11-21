# Updates all fields (table of contents, lists of figures and tables, page numbers) in the DOCX files
# produced by build_docs.js, saves them, and exports each one to PDF with Microsoft Word.
#
#   powershell -ExecutionPolicy Bypass -File docs/academic/export_pdf.ps1            # all documents
#   powershell -ExecutionPolicy Bypass -File docs/academic/export_pdf.ps1 -Only Paper

param(
    [string]$Only = ""
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$files = Get-ChildItem -Path $here -Filter *.docx | Where-Object { $_.Name -notlike "~$*" }
if ($Only) { $files = $files | Where-Object { $_.Name -like "*$Only*" } }
if (-not $files) { Write-Host "No DOCX files found in $here"; exit 1 }

$word = New-Object -ComObject Word.Application
$createdHere = ($word.Documents.Count -eq 0)
$word.Visible = $false
$word.DisplayAlerts = 0
$word.Options.CheckGrammarAndSpellingAsYouType = $false

try {
    foreach ($f in $files) {
        $docx = $f.FullName
        $pdf = [System.IO.Path]::ChangeExtension($docx, ".pdf")
        Write-Host "updating $($f.Name)"
        $doc = $word.Documents.Open($docx, $false, $false, $false)
        try {
            # Two passes: filling the table of contents can move later front-matter pages.
            for ($pass = 0; $pass -lt 2; $pass++) {
                $null = $doc.Fields.Update()
                foreach ($t in $doc.TablesOfContents) { $null = $t.Update() }
                foreach ($t in $doc.TablesOfFigures) { $null = $t.Update() }
                $doc.Repaginate()
            }
            foreach ($t in $doc.TablesOfContents) { $null = $t.UpdatePageNumbers() }
            $doc.Save()
            # 17 = wdExportFormatPDF; 0 = print quality; 1 = bookmarks from headings
            $doc.ExportAsFixedFormat($pdf, 17, $false, 0, 0, 1, 1, 0, $true, $true, 1, $true, $true, $false)
            $pages = $doc.ComputeStatistics(2)
            Write-Host ("  -> {0} ({1} pages)" -f [System.IO.Path]::GetFileName($pdf), $pages)
        }
        finally {
            $doc.Close($false)
            [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc)
        }
    }
}
finally {
    if ($createdHere) { $word.Quit() }
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
