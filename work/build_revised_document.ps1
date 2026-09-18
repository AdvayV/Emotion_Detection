$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$sourceDocument = 'C:\Users\Advay\Downloads\Proposed_Improvements_Hinglish_Emotion_Detection_Editable.docx'
$outputDirectory = Join-Path (Get-Location) 'outputs'
$outputDocument = Join-Path $outputDirectory 'Proposed_Improvements_Hinglish_Emotion_Detection_Revised_Local.docx'

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
Copy-Item -LiteralPath $sourceDocument -Destination $outputDocument -Force

function Escape-Xml([string]$Text) {
    if ($null -eq $Text) { return '' }
    return [System.Security.SecurityElement]::Escape($Text)
}

function New-Run {
    param(
        [string]$Text,
        [bool]$Bold = $false,
        [string]$Color = '263238',
        [int]$Size = 22,
        [bool]$Italic = $false
    )
    $properties = "<w:rFonts w:ascii=`"Aptos`" w:hAnsi=`"Aptos`" w:cs=`"Aptos`"/><w:color w:val=`"$Color`"/><w:sz w:val=`"$Size`"/><w:szCs w:val=`"$Size`"/>"
    if ($Bold) { $properties += '<w:b/><w:bCs/>' }
    if ($Italic) { $properties += '<w:i/><w:iCs/>' }
    return "<w:r><w:rPr>$properties</w:rPr><w:t xml:space=`"preserve`">$(Escape-Xml $Text)</w:t></w:r>"
}

function New-Paragraph {
    param(
        [string]$Text = '',
        [ValidateSet('left','center','right','both')][string]$Align = 'left',
        [int]$Size = 22,
        [bool]$Bold = $false,
        [bool]$Italic = $false,
        [string]$Color = '263238',
        [int]$Before = 0,
        [int]$After = 120,
        [int]$Line = 300,
        [bool]$KeepNext = $false,
        [bool]$PageBreakBefore = $false,
        [int]$LeftIndent = 0,
        [int]$FirstLineIndent = 0
    )
    $keep = if ($KeepNext) { '<w:keepNext/>' } else { '' }
    $pageBreak = if ($PageBreakBefore) { '<w:pageBreakBefore/>' } else { '' }
    $indent = if ($LeftIndent -gt 0 -or $FirstLineIndent -gt 0) { "<w:ind w:left=`"$LeftIndent`" w:firstLine=`"$FirstLineIndent`"/>" } else { '' }
    $pPr = "<w:pPr>$keep$pageBreak<w:jc w:val=`"$Align`"/><w:spacing w:before=`"$Before`" w:after=`"$After`" w:line=`"$Line`" w:lineRule=`"auto`"/>$indent</w:pPr>"
    return "<w:p>$pPr$(New-Run -Text $Text -Bold $Bold -Color $Color -Size $Size -Italic $Italic)</w:p>"
}

function New-RichParagraph {
    param(
        [array]$Runs,
        [ValidateSet('left','center','right','both')][string]$Align = 'left',
        [int]$Before = 0,
        [int]$After = 120,
        [int]$Line = 300,
        [bool]$KeepNext = $false
    )
    $keep = if ($KeepNext) { '<w:keepNext/>' } else { '' }
    $runXml = foreach ($run in $Runs) {
        New-Run -Text $run.Text -Bold ([bool]$run.Bold) -Color $(if ($run.Color) { $run.Color } else { '263238' }) -Size $(if ($run.Size) { $run.Size } else { 22 }) -Italic ([bool]$run.Italic)
    }
    return "<w:p><w:pPr>$keep<w:jc w:val=`"$Align`"/><w:spacing w:before=`"$Before`" w:after=`"$After`" w:line=`"$Line`" w:lineRule=`"auto`"/></w:pPr>$($runXml -join '')</w:p>"
}

function New-Heading {
    param([string]$Text, [int]$Level = 1, [bool]$PageBreakBefore = $false)
    $size = if ($Level -eq 1) { 32 } elseif ($Level -eq 2) { 27 } else { 23 }
    $color = if ($Level -eq 1) { '173F5F' } elseif ($Level -eq 2) { '20639B' } else { '3A506B' }
    $before = if ($Level -eq 1) { 300 } elseif ($Level -eq 2) { 220 } else { 160 }
    $after = if ($Level -eq 1) { 140 } else { 100 }
    return New-Paragraph -Text $Text -Size $size -Bold $true -Color $color -Before $before -After $after -KeepNext $true -PageBreakBefore $PageBreakBefore
}

function New-Bullet {
    param([string]$Text)
    return New-Paragraph -Text "•  $Text" -Size 21 -After 55 -Line 280 -LeftIndent 360
}

function New-Cell {
    param(
        [string]$Text,
        [int]$Width,
        [string]$Fill = 'FFFFFF',
        [bool]$Bold = $false,
        [string]$Color = '263238',
        [string]$Align = 'left',
        [int]$Size = 20
    )
    $cellText = New-Paragraph -Text $Text -Align $Align -Size $Size -Bold $Bold -Color $Color -After 30 -Line 260
    return @"
<w:tc>
  <w:tcPr>
    <w:tcW w:w="$Width" w:type="dxa"/>
    <w:shd w:val="clear" w:color="auto" w:fill="$Fill"/>
    <w:vAlign w:val="center"/>
    <w:tcMar><w:top w:w="100" w:type="dxa"/><w:left w:w="120" w:type="dxa"/><w:bottom w:w="100" w:type="dxa"/><w:right w:w="120" w:type="dxa"/></w:tcMar>
  </w:tcPr>
  $cellText
</w:tc>
"@
}

function New-Table {
    param(
        [array]$Headers,
        [array]$Rows,
        [array]$Widths,
        [string]$HeaderFill = '173F5F',
        [string]$AccentFill = 'EDF4F8'
    )
    $grid = ($Widths | ForEach-Object { "<w:gridCol w:w=`"$_`"/>" }) -join ''
    $headerCells = for ($i = 0; $i -lt $Headers.Count; $i++) {
        New-Cell -Text $Headers[$i] -Width $Widths[$i] -Fill $HeaderFill -Bold $true -Color 'FFFFFF' -Align 'center' -Size 19
    }
    $rowXml = @("<w:tr>$($headerCells -join '')</w:tr>")
    for ($r = 0; $r -lt $Rows.Count; $r++) {
        $fill = if ($r % 2 -eq 0) { 'FFFFFF' } else { $AccentFill }
        $cells = for ($c = 0; $c -lt $Headers.Count; $c++) {
            New-Cell -Text ([string]$Rows[$r][$c]) -Width $Widths[$c] -Fill $fill -Size 19
        }
        $rowXml += "<w:tr>$($cells -join '')</w:tr>"
    }
    return @"
<w:tbl>
  <w:tblPr>
    <w:tblW w:w="0" w:type="auto"/>
    <w:jc w:val="center"/>
    <w:tblBorders>
      <w:top w:val="single" w:sz="6" w:color="AAB7C4"/>
      <w:left w:val="single" w:sz="6" w:color="AAB7C4"/>
      <w:bottom w:val="single" w:sz="6" w:color="AAB7C4"/>
      <w:right w:val="single" w:sz="6" w:color="AAB7C4"/>
      <w:insideH w:val="single" w:sz="4" w:color="CBD5DE"/>
      <w:insideV w:val="single" w:sz="4" w:color="CBD5DE"/>
    </w:tblBorders>
    <w:tblCellMar><w:top w:w="80" w:type="dxa"/><w:left w:w="80" w:type="dxa"/><w:bottom w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar>
  </w:tblPr>
  <w:tblGrid>$grid</w:tblGrid>
  $($rowXml -join "`n")
</w:tbl>
"@
}

function New-FlowBox {
    param([string]$Text, [string]$Fill = 'E8F1F8', [string]$Border = '20639B', [int]$Width = 7800)
    $cell = New-Cell -Text $Text -Width $Width -Fill $Fill -Bold $true -Color '173F5F' -Align 'center' -Size 20
    return @"
<w:tbl>
  <w:tblPr>
    <w:tblW w:w="$Width" w:type="dxa"/>
    <w:jc w:val="center"/>
    <w:tblBorders>
      <w:top w:val="single" w:sz="12" w:color="$Border"/>
      <w:left w:val="single" w:sz="12" w:color="$Border"/>
      <w:bottom w:val="single" w:sz="12" w:color="$Border"/>
      <w:right w:val="single" w:sz="12" w:color="$Border"/>
      <w:insideH w:val="nil"/><w:insideV w:val="nil"/>
    </w:tblBorders>
  </w:tblPr>
  <w:tblGrid><w:gridCol w:w="$Width"/></w:tblGrid>
  <w:tr>$cell</w:tr>
</w:tbl>
"@
}

function New-Arrow {
    param([string]$Label = '↓')
    return New-Paragraph -Text $Label -Align center -Size 25 -Bold $true -Color '20639B' -Before 25 -After 25 -Line 240
}

function New-BranchFlow {
    param([array]$Texts, [array]$Widths, [string]$Fill = 'F2F7FA')
    $cells = for ($i = 0; $i -lt $Texts.Count; $i++) {
        New-Cell -Text $Texts[$i] -Width $Widths[$i] -Fill $Fill -Bold $true -Color '173F5F' -Align 'center' -Size 19
    }
    $grid = ($Widths | ForEach-Object { "<w:gridCol w:w=`"$_`"/>" }) -join ''
    return @"
<w:tbl>
  <w:tblPr>
    <w:tblW w:w="0" w:type="auto"/><w:jc w:val="center"/>
    <w:tblBorders>
      <w:top w:val="single" w:sz="8" w:color="6FA3C4"/><w:left w:val="single" w:sz="8" w:color="6FA3C4"/>
      <w:bottom w:val="single" w:sz="8" w:color="6FA3C4"/><w:right w:val="single" w:sz="8" w:color="6FA3C4"/>
      <w:insideV w:val="single" w:sz="8" w:color="6FA3C4"/><w:insideH w:val="nil"/>
    </w:tblBorders>
  </w:tblPr>
  <w:tblGrid>$grid</w:tblGrid>
  <w:tr>$($cells -join '')</w:tr>
</w:tbl>
"@
}

function New-Callout {
    param([string]$Title, [string]$Text, [string]$Fill = 'FFF7E6', [string]$Border = 'F0A202')
    $titleP = New-Paragraph -Text $Title -Size 21 -Bold $true -Color '805400' -After 40 -KeepNext $true
    $textP = New-Paragraph -Text $Text -Size 20 -Color '4B3A13' -After 40 -Line 275
    return @"
<w:tbl>
  <w:tblPr><w:tblW w:w="8600" w:type="dxa"/><w:jc w:val="center"/>
    <w:tblBorders><w:left w:val="single" w:sz="24" w:color="$Border"/><w:top w:val="nil"/><w:right w:val="nil"/><w:bottom w:val="nil"/></w:tblBorders>
  </w:tblPr>
  <w:tblGrid><w:gridCol w:w="8600"/></w:tblGrid>
  <w:tr><w:tc><w:tcPr><w:tcW w:w="8600" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="$Fill"/><w:tcMar><w:top w:w="140" w:type="dxa"/><w:left w:w="180" w:type="dxa"/><w:bottom w:w="140" w:type="dxa"/><w:right w:w="180" w:type="dxa"/></w:tcMar></w:tcPr>$titleP$textP</w:tc></w:tr>
</w:tbl>
"@
}

$body = [System.Collections.Generic.List[string]]::new()

# Title page
$body.Add((New-Paragraph -Text 'PROPOSED IMPROVEMENTS' -Align center -Size 20 -Bold $true -Color '20639B' -Before 500 -After 100))
$body.Add((New-Paragraph -Text 'English–Hinglish Emotion Detection' -Align center -Size 42 -Bold $true -Color '173F5F' -After 120 -Line 420))
$body.Add((New-Paragraph -Text 'A locally deployable revision of the BERT/RoBERTa + Emoji2Vec approach' -Align center -Size 24 -Color '4F6673' -After 360 -Line 320))
$body.Add((New-FlowBox -Text 'Five focused changes: Hinglish-aware language modelling • spelling robustness • contextual emoji and sarcasm analysis • local small-LLM review • stronger evaluation' -Fill 'E8F1F8' -Border '20639B' -Width 8500))
$body.Add((New-Paragraph -Text 'Revision document' -Align center -Size 20 -Italic $true -Color '637681' -Before 650 -After 80))
$body.Add((New-Paragraph -Text 'All diagrams in this document are made from editable Word tables and text.' -Align center -Size 18 -Color '637681' -After 100))
$body.Add('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

$body.Add((New-Heading -Text 'Purpose of this revision' -Level 1))
$body.Add((New-Paragraph -Text 'The original project has a useful central idea: sentiment in informal English–Hinglish messages is expressed through both words and emojis. Its English experiments are strong, but the Hinglish results show that the current system is not yet handling code-mixed language reliably. This revision keeps the original direction and concentrates on the parts that need the most work.' -Align both))
$body.Add((New-Paragraph -Text 'The proposed system is intentionally practical. BERT and RoBERTa remain as reproducible baselines. We then test Hinglish-aware encoders, preserve emotional spelling cues, model emoji meaning in context, introduce explicit sarcasm analysis, and use a small local language model only when the main classifier is uncertain. Every addition is evaluated separately before it is included in the final system.' -Align both))
$body.Add((New-Callout -Title 'Research position' -Text 'These are proposed improvements, not reported results. The final paper should claim an improvement only after the models have been trained and compared under the same data split and evaluation settings.'))

$body.Add((New-Heading -Text '1. What the current project contains' -Level 1))
$body.Add((New-Paragraph -Text 'The current paper describes an English–Hinglish sentiment system based on BERT, RoBERTa and Emoji2Vec. Text is cleaned and tokenized, Hinglish words are described as being transliterated or mapped to English equivalents, and emojis are mapped to vector representations. The text and emoji vectors are then concatenated and used for sentiment prediction.' -Align both))

$body.Add((New-Heading -Text 'Current workflow' -Level 2))
$body.Add((New-FlowBox -Text 'English or Hinglish sentence with optional emoji' -Fill 'EAF4EC' -Border '4B8B5A'))
$body.Add((New-Arrow))
$body.Add((New-FlowBox -Text 'Basic cleaning, tokenization and described Hinglish word replacement'))
$body.Add((New-Arrow))
$body.Add((New-BranchFlow -Texts @('BERT or RoBERTa text representation','Emoji2Vec representation') -Widths @(3900,3900)))
$body.Add((New-Arrow -Label '↘     concatenate     ↙'))
$body.Add((New-FlowBox -Text 'Fused feature vector and three-class classifier' -Fill 'F3ECFA' -Border '7A5195'))
$body.Add((New-Arrow))
$body.Add((New-FlowBox -Text 'Positive • Neutral • Negative' -Fill 'FFF3E6' -Border 'D97904'))
$body.Add((New-Arrow))
$body.Add((New-FlowBox -Text 'Confidence threshold converted into strong/weak intensity wording' -Fill 'FFF8E1' -Border 'D6A800'))

$body.Add((New-Heading -Text 'What the existing results tell us' -Level 2))
$body.Add((New-Table -Headers @('Model','Dataset','Reported validation accuracy') -Rows @(
    @('BERT','English','95.7%'),
    @('RoBERTa','English','95.7%'),
    @('BERT','Hinglish','59.5%'),
    @('RoBERTa','Hinglish','57.9%')
) -Widths @(3000,3000,2700)))
$body.Add((New-Paragraph -Text 'The English-to-Hinglish drop is the clearest reason for revising the system. In the reported Hinglish confusion matrices, BERT correctly identifies only 15 negative examples, while RoBERTa identifies none of the negative examples correctly. The BERT Hinglish loss curves also show training loss falling while validation loss rises beyond 1.3, which is consistent with overfitting.' -Align both -Before 140))
$body.Add((New-Paragraph -Text 'The paper describes a complete Emoji2Vec fusion and Hinglish transliteration pipeline, but the displayed training code mainly fine-tunes standard sequence-classification models. The revised implementation should therefore make each claimed component visible in the code and test it through an ablation study.' -Align both))

$body.Add((New-Heading -Text '2. Current system versus proposed system' -Level 1 -PageBreakBefore $true))
$body.Add((New-Table -Headers @('Area','Current approach','Proposed revision') -Rows @(
    @('Language representation','English-oriented BERT and RoBERTa','Retain both as baselines; compare MuRIL, mBERT and XLM-R under the same setup'),
    @('Hinglish spelling','Basic replacement such as acha → good','Canonical Hinglish forms, phonetic variants, elongation features and consistency training'),
    @('Emoji and sarcasm','Static Emoji2Vec concatenation; no explicit sarcasm task','Contextual text–emoji fusion with sentiment and sarcasm prediction branches'),
    @('Difficult cases','One classifier handles every example','A local Qwen3 4B reviewer handles only uncertain or conflicting examples'),
    @('Training and evidence','Ordinary cross-entropy, three epochs and emphasis on accuracy','Class-aware training when justified, early stopping, macro-F1, per-class recall, calibration and ablation')
) -Widths @(1900,3350,3650)))

$body.Add((New-Heading -Text 'Proposed end-to-end workflow' -Level 2))
$body.Add((New-FlowBox -Text 'Raw English–Hinglish message + emoji' -Fill 'EAF4EC' -Border '4B8B5A'))
$body.Add((New-Arrow))
$body.Add((New-FlowBox -Text 'Local Hinglish normalizer: spelling variants, repeated letters, negation and language cues'))
$body.Add((New-Arrow))
$body.Add((New-BranchFlow -Texts @('Multilingual/Indic text encoder','Contextual emoji representation') -Widths @(3900,3900)))
$body.Add((New-Arrow -Label '↘     gated fusion     ↙'))
$body.Add((New-BranchFlow -Texts @('Sentiment prediction','Sarcasm and conflict prediction') -Widths @(3900,3900) -Fill 'F3ECFA'))
$body.Add((New-Arrow))
$body.Add((New-FlowBox -Text 'Confidence router: is the prediction clear and internally consistent?' -Fill 'FFF3E6' -Border 'D97904'))
$body.Add((New-Arrow -Label '↙ clear                         uncertain ↘'))
$body.Add((New-BranchFlow -Texts @('Accept main classifier result','Review locally with Qwen3 4B') -Widths @(3900,3900) -Fill 'FFF8E1'))
$body.Add((New-Arrow -Label '↘                               ↙'))
$body.Add((New-FlowBox -Text 'Final sentiment, sarcasm flag and concise supporting evidence' -Fill 'EAF4EC' -Border '4B8B5A'))

$body.Add((New-Heading -Text '3. The five proposed changes' -Level 1 -PageBreakBefore $true))

$body.Add((New-Heading -Text 'Change 1 — Start with a Hinglish-aware encoder' -Level 2))
$body.Add((New-Paragraph -Text 'The current implementation applies bert-base-uncased and roberta-base to Romanized Hindi even though both checkpoints are primarily English-oriented. We will retain them because they provide the baseline reported in the original paper. The proposed experiment will then compare mBERT, MuRIL and XLM-R using the same train/validation split, preprocessing and evaluation procedure.' -Align both))
$body.Add((New-Paragraph -Text 'MuRIL is the most directly relevant starting point because it was pretrained on Indian languages and transliterated counterparts. XLM-R provides a strong multilingual comparison, while mBERT provides a simpler multilingual baseline. The model will be selected from experimental results rather than assumed to be better in advance.' -Align both))
$body.Add((New-Table -Headers @('Model','Role in the revised project','Reason for inclusion') -Rows @(
    @('BERT','Original baseline','Matches the current implementation'),
    @('RoBERTa','Original baseline','Tests whether stronger English pretraining helps'),
    @('mBERT','Multilingual baseline','Broad multilingual BERT comparison'),
    @('MuRIL','Primary Indic candidate','Indian languages plus transliterated training data'),
    @('XLM-R','Strong multilingual candidate','Large multilingual pretraining corpus')
) -Widths @(1800,3100,3800)))
$body.Add((New-Callout -Title 'Direct connection to the paper' -Text 'This change targets the reported fall from 95.7% English accuracy to 59.5% and 57.9% on Hinglish, while preserving the original BERT/RoBERTa experiments for comparison.' -Fill 'EEF7F0' -Border '4B8B5A'))

$body.Add((New-Heading -Text 'Change 2 — Normalize Hinglish spelling without losing emotional emphasis' -Level 2))
$body.Add((New-Paragraph -Text 'Romanized Hindi has no single everyday spelling standard. A word may appear as acha, accha, achha or acchaaa. A blanket rule that removes every repeated character is unsafe: goooood should become good, not god. The proposed normalizer therefore combines repetition detection, candidate generation, a local Hinglish variant lexicon, phonetic similarity and sentence context.' -Align both))
$body.Add((New-Paragraph -Text 'The system will preserve both the raw sentence and its normalized form. When extra letters express emphasis, it will also store an elongation feature. This allows the encoder to see a stable word such as accha while the emotion layer still knows that the user wrote acchaaa.' -Align both))

$body.Add((New-Heading -Text 'Editable normalization flow' -Level 3))
$body.Add((New-FlowBox -Text 'Raw token: acchaaa' -Fill 'FFF3E6' -Border 'D97904' -Width 6500))
$body.Add((New-Arrow))
$body.Add((New-FlowBox -Text 'Detect expressive repetition: aaa' -Width 6500))
$body.Add((New-Arrow))
$body.Add((New-FlowBox -Text 'Generate candidates and compare with the local Hinglish lexicon' -Width 6500))
$body.Add((New-Arrow))
$body.Add((New-FlowBox -Text 'Use phonetic similarity and sentence context to choose accha' -Width 6500))
$body.Add((New-Arrow))
$body.Add((New-BranchFlow -Texts @('Normalized token: accha','Preserved feature: elongated = yes') -Widths @(3250,3250) -Fill 'EAF4EC'))

$body.Add((New-Table -Headers @('Observed form','Canonical form','Additional information') -Rows @(
    @('acha / achha / accha','accha','Known spelling variant'),
    @('acchaaa','accha','Elongated final vowel'),
    @('bohot / bahut','bahut','Phonetic spelling variant'),
    @('bohottt','bahut','Elongated final consonant'),
    @('nhi / nahin','nahi','Negation; must be preserved'),
    @('goooood','good','Candidate selection prevents god')
) -Widths @(2600,2500,3600)))
$body.Add((New-Paragraph -Text 'Training will include meaning-preserving spelling variants so that equivalent forms receive consistent sentiment predictions. Meaning-changing variants, especially those containing nahi, lekin or contradictory emojis, will be kept separate. This connects normalization with the counterfactual consistency test without turning it into an unrelated sixth module.' -Align both -Before 140))

$body.Add((New-Heading -Text 'Change 3 — Model emoji context and sarcasm together' -Level 2 -PageBreakBefore $true))
$body.Add((New-Paragraph -Text 'The original design concatenates the text vector and Emoji2Vec vector. Concatenation supplies both signals to the classifier, but it does not explicitly require the system to explain whether the emoji supports, weakens or contradicts the surrounding words. We propose a small gated-fusion or cross-attention layer that learns the contribution of the emoji for each message.' -Align both))
$body.Add((New-Paragraph -Text 'Sarcasm is placed in the same improvement because it often appears through an incongruity between literal wording and context. In “wah, kya service hai, 2 ghante late 😂”, positive-looking words occur beside an obviously negative situation. A shared multilingual representation will feed a sentiment head and a sarcasm head. The sarcasm signal can then assist the final sentiment decision.' -Align both))
$body.Add((New-BranchFlow -Texts @('Text encoder: wording, negation and context','Emoji encoder: emoji meaning and sequence') -Widths @(4100,4100) -Fill 'E8F1F8'))
$body.Add((New-Arrow -Label '↘             contextual fusion             ↙'))
$body.Add((New-FlowBox -Text 'Shared representation of the message' -Fill 'F3ECFA' -Border '7A5195'))
$body.Add((New-Arrow -Label '↙                                      ↘'))
$body.Add((New-BranchFlow -Texts @('Sentiment head: positive / neutral / negative','Sarcasm head: sarcastic / non-sarcastic') -Widths @(4100,4100) -Fill 'FFF8E1'))
$body.Add((New-Paragraph -Text 'This component requires labelled sarcasm examples. If the original dataset has no sarcasm labels, we will create a smaller manually reviewed subset or use a suitable public Hinglish sarcasm dataset. Emoji rules alone will not be presented as sarcasm detection.' -Align both -Before 140))

$body.Add((New-Heading -Text 'Change 4 — Add a free, local small-language-model reviewer' -Level 2))
$body.Add((New-Paragraph -Text 'A small language model can help with examples that need reasoning over negation, sarcasm or text–emoji conflict. It should not replace the main classifier or process every sentence. Instead, a confidence router will send only uncertain examples to a locally running Qwen3 4B model.' -Align both))
$body.Add((New-Paragraph -Text 'The proposed checkpoint is Qwen3-4B-Instruct-2507 or its compatible quantized local package. It is available under the Apache 2.0 licence and can run through an open local runtime such as Ollama or llama.cpp. After the initial download, inference does not require a paid service or cloud API. User messages and predictions remain on the computer.' -Align both))

$body.Add((New-Heading -Text 'Local review decision flow' -Level 3))
$body.Add((New-FlowBox -Text 'MuRIL/XLM-R produces sentiment probabilities' -Width 7000))
$body.Add((New-Arrow))
$body.Add((New-FlowBox -Text 'Check confidence, sarcasm score, negation and text–emoji agreement' -Fill 'FFF3E6' -Border 'D97904' -Width 7000))
$body.Add((New-Arrow -Label '↙ clear result                         conflict or uncertainty ↘'))
$body.Add((New-BranchFlow -Texts @('Return main classifier result','Ask local Qwen3 4B for a constrained review') -Widths @(3900,3900) -Fill 'FFF8E1'))
$body.Add((New-Arrow -Label '                                      ↓'))
$body.Add((New-BranchFlow -Texts @('','Sentiment • sarcasm • emoji relation • evidence span') -Widths @(3900,3900) -Fill 'F3ECFA'))
$body.Add((New-Arrow -Label '↘                                   ↙'))
$body.Add((New-FlowBox -Text 'Combine results using a validation-tuned decision rule' -Fill 'EAF4EC' -Border '4B8B5A' -Width 7000))

$body.Add((New-Table -Headers @('Local component','Purpose','Approximate local requirement') -Rows @(
    @('MuRIL','Primary Hinglish encoder','About 1 GB of published model weights'),
    @('Qwen3 4B, 4-bit','Uncertain-case reviewer','About 2.5–3 GB model package; additional runtime memory required'),
    @('Qwen3 1.7B, 4-bit','Lower-memory alternative','About 1.4 GB model package, with lower expected capability'),
    @('Normalizer and lexicon','Spelling and elongation handling','Small local code and data files')
) -Widths @(2500,3550,2850)))
$body.Add((New-Callout -Title 'Scope of the LLM' -Text 'The LLM will return only a fixed local schema: sentiment label, sarcasm flag, text–emoji relation and a short evidence span. Invalid or unsupported output will be rejected. This keeps the experiment measurable and prevents free-form generation from becoming the classifier.' -Fill 'EEF7F0' -Border '4B8B5A'))

$body.Add((New-Heading -Text 'Change 5 — Improve training and make the evidence stronger' -Level 2 -PageBreakBefore $true))
$body.Add((New-Paragraph -Text 'The revised training process will first inspect the class distribution and confusion matrices. If the negative class is under-represented, class-weighted cross-entropy, focal loss or balanced sampling can be compared. Early stopping and model selection based on validation macro-F1 will help reduce overfitting. The choice of balancing method will be supported by the observed data rather than assumed from the zero-negative result alone.' -Align both))
$body.Add((New-Paragraph -Text 'Accuracy will remain in the report, but it will not be the main measure for Hinglish. Macro-F1 gives equal importance to all three classes, while per-class recall reveals whether negative examples are still being ignored. Calibration will show whether confidence values are trustworthy enough to drive the local-LLM router.' -Align both))

$body.Add((New-Heading -Text 'Ablation and evaluation sequence' -Level 3))
$body.Add((New-FlowBox -Text 'Reproduce BERT and RoBERTa baselines' -Width 7000))
$body.Add((New-Arrow))
$body.Add((New-FlowBox -Text 'Add multilingual/Indic encoder' -Width 7000))
$body.Add((New-Arrow))
$body.Add((New-FlowBox -Text 'Add spelling normalization and elongation feature' -Width 7000))
$body.Add((New-Arrow))
$body.Add((New-FlowBox -Text 'Add contextual emoji fusion and sarcasm head' -Width 7000))
$body.Add((New-Arrow))
$body.Add((New-FlowBox -Text 'Add local Qwen reviewer for routed cases' -Width 7000))
$body.Add((New-Arrow))
$body.Add((New-FlowBox -Text 'Compare full model with every earlier stage' -Fill 'EAF4EC' -Border '4B8B5A' -Width 7000))

$body.Add((New-Table -Headers @('Measure','What it answers') -Rows @(
    @('Accuracy','How often is the overall label correct?'),
    @('Macro-F1','Does the system perform reasonably across all three classes?'),
    @('Per-class recall','Can it recognise negative, neutral and positive examples separately?'),
    @('Sarcasm F1','Does the additional branch recognise sarcastic examples?'),
    @('Variation consistency','Do acha, accha, achha and elongated forms behave consistently?'),
    @('Sentiment-flip accuracy','Does negation or contradictory context change the prediction correctly?'),
    @('Calibration error','Does the reported confidence reflect actual reliability?'),
    @('Routing coverage and latency','How often is Qwen used, and what local cost does it add?')
) -Widths @(3000,5700)))

$body.Add((New-Heading -Text '4. Proposed implementation plan' -Level 1))
$body.Add((New-Table -Headers @('Stage','Work to complete','Output') -Rows @(
    @('1. Reproduction','Run the original BERT and RoBERTa experiments using a fixed split and seed','Verified baseline metrics and confusion matrices'),
    @('2. Language backbone','Fine-tune and compare mBERT, MuRIL and XLM-R','Selected Hinglish encoder'),
    @('3. Normalization','Build the variant lexicon, elongation detector and consistency pairs','Local preprocessing module and robustness set'),
    @('4. Context modelling','Implement gated emoji fusion and obtain sarcasm labels','Sentiment–sarcasm multi-task model'),
    @('5. Local review','Run quantized Qwen3 4B locally and tune the routing rule','Hybrid local inference pipeline'),
    @('6. Evaluation','Run ablations, error analysis, calibration and timing','Evidence for each proposed contribution')
) -Widths @(1900,4550,2450)))

$body.Add((New-Heading -Text '5. Expected contribution' -Level 1))
$body.Add((New-Paragraph -Text 'The revised project will remain recognisably connected to the original paper: it still studies lexical context, emojis and English–Hinglish sentiment. Its contribution becomes clearer because each weakness in the original results is matched with a testable change. Multilingual pretraining addresses the linguistic mismatch; the normalizer handles unstable Roman spellings; contextual fusion and sarcasm analysis handle disagreement between literal words and emojis; the local small language model reviews difficult cases; and the evaluation plan checks whether these additions improve the under-performing classes.' -Align both))
$body.Add((New-Paragraph -Text 'If the experiments support it, the final system can be described as a locally deployable, Hinglish-aware sentiment model that preserves expressive spelling, interprets emojis in context and uses a small open model selectively for ambiguous input. The research value will come from the controlled comparisons and error analysis, not from the number of components alone.' -Align both))

$body.Add((New-Callout -Title 'Proposed final statement' -Text 'We propose a fully local English–Hinglish sentiment framework in which an Indic-aware encoder performs the main classification, spelling normalization preserves expressive elongation, text and emoji cues are fused contextually, and a compact open language model reviews only uncertain or sarcastic cases.'))

$body.Add((New-Heading -Text 'References' -Level 1 -PageBreakBefore $true))
$body.Add((New-Paragraph -Text '[1] Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT.' -Size 19 -After 80))
$body.Add((New-Paragraph -Text '[2] Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.' -Size 19 -After 80))
$body.Add((New-Paragraph -Text '[3] Khanuja, S. et al. (2021). MuRIL: Multilingual Representations for Indian Languages. arXiv:2103.10730.' -Size 19 -After 80))
$body.Add((New-Paragraph -Text '[4] Conneau, A. et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. ACL.' -Size 19 -After 80))
$body.Add((New-Paragraph -Text '[5] Joshi, A., Prabhu, A., Shrivastava, M. and Varma, V. (2016). Towards Sub-Word Level Compositions for Sentiment Analysis of Hindi-English Code Mixed Text. COLING.' -Size 19 -After 80))
$body.Add((New-Paragraph -Text '[6] Qwen Team (2025). Qwen3 Technical Report and Qwen3-4B-Instruct-2507 model release. Model licence: Apache 2.0. https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507' -Size 19 -After 80))
$body.Add((New-Paragraph -Text '[7] Google Research. MuRIL model card and checkpoint. Model licence: Apache 2.0. https://huggingface.co/google/muril-base-cased' -Size 19 -After 80))
$body.Add((New-Paragraph -Text '[8] Ollama. Qwen3 local quantized model packages. https://ollama.com/library/qwen3/tags' -Size 19 -After 80))

$sectionProperties = @"
<w:sectPr>
  <w:pgSz w:w="12240" w:h="15840"/>
  <w:pgMar w:top="900" w:right="1080" w:bottom="900" w:left="1080" w:header="450" w:footer="450" w:gutter="0"/>
  <w:cols w:space="720"/>
  <w:docGrid w:linePitch="360"/>
</w:sectPr>
"@

$documentXml = @"
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>
    $($body -join "`n")
    $sectionProperties
  </w:body>
</w:document>
"@

$stream = [System.IO.File]::Open($outputDocument, [System.IO.FileMode]::Open, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
$archive = [System.IO.Compression.ZipArchive]::new($stream, [System.IO.Compression.ZipArchiveMode]::Update, $false)
try {
    $oldEntry = $archive.GetEntry('word/document.xml')
    if ($null -ne $oldEntry) { $oldEntry.Delete() }
    $newEntry = $archive.CreateEntry('word/document.xml', [System.IO.Compression.CompressionLevel]::Optimal)
    $entryStream = $newEntry.Open()
    try {
        $writer = [System.IO.StreamWriter]::new($entryStream, [System.Text.UTF8Encoding]::new($false))
        try { $writer.Write($documentXml) } finally { $writer.Dispose() }
    } finally { $entryStream.Dispose() }

    $coreEntry = $archive.GetEntry('docProps/core.xml')
    if ($null -ne $coreEntry) {
        $coreEntry.Delete()
        $coreEntry = $archive.CreateEntry('docProps/core.xml', [System.IO.Compression.CompressionLevel]::Optimal)
        $coreXml = @"
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Proposed Improvements to English–Hinglish Emotion Detection</dc:title>
  <dc:subject>Local Hinglish sentiment, emoji and sarcasm analysis</dc:subject>
  <dc:creator>Research Project Team</dc:creator>
  <cp:keywords>Hinglish; sentiment analysis; MuRIL; Qwen; sarcasm; emoji</cp:keywords>
  <dc:description>Revised proposal with five focused improvements and editable Word-native flow diagrams.</dc:description>
  <cp:lastModifiedBy>Research Project Team</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-09-08T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-09-08T00:00:00Z</dcterms:modified>
</cp:coreProperties>
"@
        $coreStream = $coreEntry.Open()
        try {
            $coreWriter = [System.IO.StreamWriter]::new($coreStream, [System.Text.UTF8Encoding]::new($false))
            try { $coreWriter.Write($coreXml) } finally { $coreWriter.Dispose() }
        } finally { $coreStream.Dispose() }
    }
} finally {
    $archive.Dispose()
    $stream.Dispose()
}

Get-Item -LiteralPath $outputDocument | Select-Object FullName, Length, LastWriteTime
