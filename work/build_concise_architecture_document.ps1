$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$template = Join-Path (Get-Location) 'outputs\Proposed_Improvements_Hinglish_Emotion_Detection_Revised_Local.docx'
$output = Join-Path (Get-Location) 'outputs\Proposed_Improvements_Hinglish_Emotion_Detection_Concise_Architecture.docx'
Copy-Item -LiteralPath $template -Destination $output -Force

function E { param([string]$s) if ($null -eq $s) { return '' }; [System.Security.SecurityElement]::Escape($s) }
function New-RunXml {
  param([string]$t,[int]$size=19,[bool]$bold=$false,[string]$color='263238')
  $b = if($bold){'<w:b/><w:bCs/>'}else{''}
  "<w:r><w:rPr><w:rFonts w:ascii=`"Aptos`" w:hAnsi=`"Aptos`"/><w:color w:val=`"$color`"/><w:sz w:val=`"$size`"/><w:szCs w:val=`"$size`"/>$b</w:rPr><w:t xml:space=`"preserve`">$(E $t)</w:t></w:r>"
}
function P {
  param([string]$t,[string]$align='left',[int]$size=19,[bool]$bold=$false,[string]$color='263238',[int]$before=0,[int]$after=70,[bool]$keep=$false)
  $k=if($keep){'<w:keepNext/>'}else{''}
  "<w:p><w:pPr>$k<w:jc w:val=`"$align`"/><w:spacing w:before=`"$before`" w:after=`"$after`" w:line=`"240`" w:lineRule=`"auto`"/></w:pPr>$(New-RunXml -t $t -size $size -bold $bold -color $color)</w:p>"
}
function New-HeadingXml {
  param([string]$t,[int]$level=1)
  if($level -eq 1){P $t 'left' 27 $true '173F5F' 160 70 $true}else{P $t 'left' 22 $true '20639B' 110 50 $true}
}
function Cell {
  param([string]$t,[int]$w,[string]$fill='FFFFFF',[bool]$bold=$false,[string]$align='left')
  "<w:tc><w:tcPr><w:tcW w:w=`"$w`" w:type=`"dxa`"/><w:shd w:val=`"clear`" w:color=`"auto`" w:fill=`"$fill`"/><w:vAlign w:val=`"center`"/><w:tcMar><w:top w:w=`"55`" w:type=`"dxa`"/><w:left w:w=`"80`" w:type=`"dxa`"/><w:bottom w:w=`"55`" w:type=`"dxa`"/><w:right w:w=`"80`" w:type=`"dxa`"/></w:tcMar></w:tcPr>$(P $t $align 17 $bold $(if($fill -eq '173F5F'){'FFFFFF'}else{'263238'}) 0 10)</w:tc>"
}
function T {
  param([array]$headers,[array]$rows,[array]$widths)
  $grid=($widths|ForEach-Object{"<w:gridCol w:w=`"$_`"/>"})-join ''
  $h=for($i=0;$i -lt $headers.Count;$i++){Cell $headers[$i] $widths[$i] '173F5F' $true 'center'}
  $trs=@("<w:tr>$($h -join '')</w:tr>")
  for($r=0;$r -lt $rows.Count;$r++){
    $fill=if($r%2 -eq 0){'FFFFFF'}else{'EEF4F7'}
    $c=for($i=0;$i -lt $headers.Count;$i++){Cell ([string]$rows[$r][$i]) $widths[$i] $fill $false 'left'}
    $trs+="<w:tr>$($c -join '')</w:tr>"
  }
  @"
<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:jc w:val="center"/><w:tblBorders><w:top w:val="single" w:sz="5" w:color="A9BBC7"/><w:left w:val="single" w:sz="5" w:color="A9BBC7"/><w:bottom w:val="single" w:sz="5" w:color="A9BBC7"/><w:right w:val="single" w:sz="5" w:color="A9BBC7"/><w:insideH w:val="single" w:sz="3" w:color="D5E0E6"/><w:insideV w:val="single" w:sz="3" w:color="D5E0E6"/></w:tblBorders></w:tblPr><w:tblGrid>$grid</w:tblGrid>$($trs -join "`n")</w:tbl>
"@
}
function Box {
  param([string]$t,[string]$fill='E8F1F8',[string]$border='20639B',[int]$width=8600)
  $c=Cell $t $width $fill $true 'center'
  "<w:tbl><w:tblPr><w:tblW w:w=`"$width`" w:type=`"dxa`"/><w:jc w:val=`"center`"/><w:tblBorders><w:top w:val=`"single`" w:sz=`"10`" w:color=`"$border`"/><w:left w:val=`"single`" w:sz=`"10`" w:color=`"$border`"/><w:bottom w:val=`"single`" w:sz=`"10`" w:color=`"$border`"/><w:right w:val=`"single`" w:sz=`"10`" w:color=`"$border`"/></w:tblBorders></w:tblPr><w:tblGrid><w:gridCol w:w=`"$width`"/></w:tblGrid><w:tr>$c</w:tr></w:tbl>"
}
function Arrow { param([string]$text='↓') P $text 'center' 21 $true '20639B' 0 0 }
function Branch {
  param([string]$left,[string]$right,[string]$fill='F3ECFA')
  $l=Cell $left 4300 $fill $true 'center'; $r=Cell $right 4300 $fill $true 'center'
  "<w:tbl><w:tblPr><w:tblW w:w=`"0`" w:type=`"auto`"/><w:jc w:val=`"center`"/><w:tblBorders><w:top w:val=`"single`" w:sz=`"6`" w:color=`"8E6BA6`"/><w:left w:val=`"single`" w:sz=`"6`" w:color=`"8E6BA6`"/><w:bottom w:val=`"single`" w:sz=`"6`" w:color=`"8E6BA6`"/><w:right w:val=`"single`" w:sz=`"6`" w:color=`"8E6BA6`"/><w:insideV w:val=`"single`" w:sz=`"6`" w:color=`"8E6BA6`"/></w:tblBorders></w:tblPr><w:tblGrid><w:gridCol w:w=`"4300`"/><w:gridCol w:w=`"4300`"/></w:tblGrid><w:tr>$l$r</w:tr></w:tbl>"
}

$b=[Collections.Generic.List[string]]::new()
$b.Add((P 'PROPOSED IMPROVEMENTS TO ENGLISH–HINGLISH EMOTION DETECTION' 'center' 28 $true '173F5F' 120 30))
$b.Add((P 'Concise architecture and local implementation plan' 'center' 19 $false '637681' 0 110))
$b.Add((P 'All proposed models are small: BERT, RoBERTa, mBERT, MuRIL and XLM-R base are encoder models below 1B parameters; the local reviewer is Qwen3 4B. No model exceeds 8B parameters.' 'center' 16 $false '4F6673' 0 100))

$b.Add((New-HeadingXml '1. Current system and the reason for revision' 1))
$b.Add((P 'The existing paper classifies English and Hinglish messages as positive, neutral or negative. It describes a pipeline that cleans text, applies Hinglish word handling, produces a BERT or RoBERTa text representation, combines it with Emoji2Vec, and maps classifier confidence to an intensity label. The key issue is that the reported English accuracy is 95.7%, while Hinglish accuracy falls to 59.5% for BERT and 57.9% for RoBERTa.' 'both'))
$b.Add((T @('What exists now','Why it is limited') @(
  @('BERT/RoBERTa trained mainly on English','Romanized Hindi and code switching are not their main pretraining setting'),
  @('Simple text–emoji concatenation','An emoji can support, weaken or contradict the words around it'),
  @('Three-class prediction plus confidence rule','Confidence is not the same as a learned emotion-intensity class'),
  @('One classifier handles all inputs','Sarcasm and conflicting emoji cases need extra context'),
  @('Accuracy is prominent','Negative-class recall and robustness are needed for Hinglish')
) @(4300,4300)))

$b.Add((New-HeadingXml '2. Proposed architecture: how one message moves through the system' 1))
$b.Add((P 'The revised architecture has five layers. Each layer solves a specific weakness of the original system. The main encoder performs the normal sentiment task; the small language model is used only for cases that are uncertain or internally contradictory.' 'both'))
$b.Add((Box '1. Raw message: English–Hinglish text + emoji' 'EAF4EC' '4B8B5A'))
$b.Add((Arrow))
$b.Add((Box '2. Hinglish normalizer: keep the raw form, create a canonical form, preserve negation and record elongation' 'E8F1F8' '20639B'))
$b.Add((Arrow))
$b.Add((Branch '3a. Multilingual text encoder: MuRIL / mBERT / XLM-R base' '3b. Emoji encoder: emoji meaning and position in the message'))
$b.Add((Arrow '↘               contextual gated fusion               ↙'))
$b.Add((Box '4. Shared representation: combines wording, code-switching, emoji context and elongation signal' 'F3ECFA' '7A5195'))
$b.Add((Arrow '↙                                     ↘'))
$b.Add((Branch 'Sentiment head: positive / neutral / negative' 'Sarcasm head: sarcastic / non-sarcastic' 'FFF8E1'))
$b.Add((Arrow))
$b.Add((Box '5. Confidence router: clear result → return it; uncertain or conflicting result → review locally with Qwen3 4B' 'FFF3E6' 'D97904'))
$b.Add((Arrow))
$b.Add((Box 'Final output: sentiment, sarcasm flag, confidence and short evidence span' 'EAF4EC' '4B8B5A'))

$b.Add((New-HeadingXml 'How to read this architecture' 2))
$b.Add((P 'For “wah, kya service hai, 2 ghante late 😂”, the normalizer preserves the words and detects the contrast between positive-looking language and the complaint. MuRIL or another selected multilingual encoder reads the full sentence. The emoji branch contributes its contextual signal. The sarcasm head checks for inconsistency, and only if the prediction remains uncertain does local Qwen3 4B return a restricted review: sentiment, sarcasm, emoji relation and evidence such as “2 ghante late”.' 'both'))

$b.Add((New-HeadingXml '3. Five focused improvements' 1))
$b.Add((T @('Change','What is added','Connection to the existing paper') @(
  @('1. Hinglish-aware encoder','Compare MuRIL, mBERT and XLM-R base with BERT/RoBERTa baselines','Directly addresses the English-to-Hinglish performance drop'),
  @('2. Spelling and elongation normalizer','Map acha, accha, achha and acchaaa to a canonical form while preserving emphasis','Strengthens the described transliteration/preprocessing stage'),
  @('3. Contextual emoji + sarcasm','Use gated fusion and a separate sarcasm head','Improves the original Emoji2Vec + text idea for conflicting cues'),
  @('4. Local Qwen3 4B reviewer','Review only uncertain or sarcastic examples through an open local runtime','Adds reasoning without replacing the fast main classifier'),
  @('5. Stronger evaluation','Macro-F1, per-class recall, calibration, early stopping and ablations','Makes the Hinglish results more reliable and easier to defend')
) @(1800,3650,3150)))

$b.Add((New-HeadingXml 'The normalizer in simple terms' 2))
$b.Add((Box 'Raw token: acchaaa  →  detect repeated letters  →  check Hinglish vocabulary and phonetic variants  →  canonical token: accha + feature: elongated = yes' 'E8F1F8' '20639B'))
$b.Add((P 'The model receives the normalized token so that spelling noise does not split one word into many forms. It also receives an elongation feature because acchaaa, bohottt or bekaaar may express stronger feeling. Negation is never removed: achhi nahi and achhi should remain different. Unknown words remain unchanged unless the local vocabulary and context support a safe match.' 'both'))

$b.Add((New-HeadingXml '4. Local and small-model requirement' 1))
$b.Add((T @('Model','Role','Size / local use') @(
  @('BERT base','Existing baseline','About 110M parameters; local'),
  @('RoBERTa base','Existing baseline','About 125M parameters; local'),
  @('mBERT / MuRIL','Multilingual and Indic candidates','About 110M parameters; local'),
  @('XLM-R base','Multilingual comparison','About 270M parameters; local'),
  @('Qwen3 4B Instruct','Reviewer for routed difficult cases','4B parameters; Apache 2.0; local in 4-bit form')
) @(2200,3700,2700)))
$b.Add((P 'Qwen is not called for every sentence. The primary encoder handles ordinary inputs quickly. A routing rule sends a message to Qwen only when confidence is low, the sarcasm score is high, or text and emoji signals disagree. This keeps the design practical for a free local computer setup and makes the contribution testable.' 'both'))

$b.Add((New-HeadingXml '5. Evaluation plan' 1))
$b.Add((Box 'Baseline BERT/RoBERTa  →  multilingual encoder  →  normalizer  →  emoji + sarcasm layer  →  routed Qwen review  →  compare every stage' 'F3ECFA' '7A5195'))
$b.Add((T @('Measure','Question it answers') @(
  @('Macro-F1 and per-class recall','Does the model handle negative, neutral and positive classes fairly?'),
  @('Variation consistency','Do acha, accha, achha and acchaaa receive compatible predictions?'),
  @('Sentiment-flip accuracy','Does adding nahi or a contradictory emoji change the prediction correctly?'),
  @('Sarcasm F1','Does the new sarcasm branch add measurable value?'),
  @('Calibration and routing rate','Are confidence values reliable, and how often is Qwen actually needed?')
) @(3100,5500)))
$b.Add((P 'The final report should describe improvements only when the ablation results support them. This keeps the project grounded in the original paper while giving it a clearer and more defensible contribution.' 'both'))

$b.Add((New-HeadingXml 'References' 1))
$b.Add((P 'Devlin et al. (2019), BERT.  Liu et al. (2019), RoBERTa.  Khanuja et al. (2021), MuRIL.  Conneau et al. (2020), XLM-R.  Qwen Team (2025), Qwen3 4B Instruct.  Qwen3 4B and MuRIL checkpoints are published under Apache 2.0 licences.' 'left' 16 $false '4F6673' 0 0))

$sect='<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="650" w:right="720" w:bottom="650" w:left="720" w:header="300" w:footer="300" w:gutter="0"/><w:cols w:space="450"/></w:sectPr>'
$xml=@"
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:body>$($b -join "`n")$sect</w:body></w:document>
"@

$s=[IO.File]::Open($output,[IO.FileMode]::Open,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$z=[IO.Compression.ZipArchive]::new($s,[IO.Compression.ZipArchiveMode]::Update,$false)
try {
  $old=$z.GetEntry('word/document.xml'); if($null -ne $old){$old.Delete()}
  $entry=$z.CreateEntry('word/document.xml',[IO.Compression.CompressionLevel]::Optimal)
  $es=$entry.Open(); try {$w=[IO.StreamWriter]::new($es,[Text.UTF8Encoding]::new($false)); try {$w.Write($xml)} finally {$w.Dispose()}} finally {$es.Dispose()}
} finally {$z.Dispose();$s.Dispose()}

Get-Item -LiteralPath $output | Select-Object FullName,Length,LastWriteTime
