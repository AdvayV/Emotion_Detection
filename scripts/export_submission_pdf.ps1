param([string]$InputPath,[string]$OutputPath)
$taskWord=$null
$taskDocument=$null
try {
 $taskWord=New-Object -ComObject Word.Application
 $taskWord.Visible=$false
 $taskWord.DisplayAlerts=0
 $taskDocument=$taskWord.Documents.Open($InputPath,$false,$true)
 $taskDocument.Repaginate()
 $taskDocument.ExportAsFixedFormat($OutputPath,17)
 Write-Output ('Exported '+$taskDocument.ComputeStatistics(2)+' pages')
} finally {
 if($null -ne $taskDocument){$taskDocument.Close(0);[System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskDocument)|Out-Null}
 if($null -ne $taskWord){$taskWord.Quit();[System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskWord)|Out-Null}
}
