/** Render the three IEEE manuscript Mermaid sources to editable SVG and print PNG. */
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.argv[2]);
const mermaidScript = process.argv[3];
const root = path.resolve(__dirname, '..');
const sourceDir = path.join(root, 'docs/ieee_manuscript/diagrams');
const outputDir = path.join(root, 'docs/ieee_manuscript/figures');

(async () => {
  const edge = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
  const executablePath = process.env.BROWSER_EXECUTABLE || (fs.existsSync(edge) ? edge : undefined);
  const browser = await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
  const page = await browser.newPage({ viewport: { width: 1800, height: 2400 }, deviceScaleFactor: 3 });
  await page.setContent('<!doctype html><html><body style="margin:0;padding:24px;background:#fff"><div id="diagram"></div></body></html>');
  await page.addScriptTag({ path: mermaidScript });
  await page.evaluate(() => mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    theme: 'base',
    fontFamily: 'Arial',
    themeVariables: {
      background: '#ffffff', primaryColor: '#ffffff', primaryTextColor: '#000000',
      primaryBorderColor: '#000000', lineColor: '#000000', secondaryColor: '#ffffff',
      secondaryTextColor: '#000000', secondaryBorderColor: '#000000', tertiaryColor: '#ffffff',
      tertiaryTextColor: '#000000', tertiaryBorderColor: '#000000', clusterBkg: '#ffffff',
      clusterBorder: '#000000', edgeLabelBackground: '#ffffff', fontSize: '18px',
    },
    flowchart: { htmlLabels: false, nodeSpacing: 28, rankSpacing: 34, padding: 16, curve: 'linear', useMaxWidth: false },
  }));
  fs.mkdirSync(outputDir, { recursive: true });
  const names = ['master_architecture.mmd', 'prediction_review.mmd', 'training_review_cycle.mmd'];
  for (const name of names) {
    const source = fs.readFileSync(path.join(sourceDir, name), 'utf8');
    const svg = await page.evaluate(async code => (await mermaid.render('ieee_diagram', code)).svg, source);
    await page.evaluate(markup => { document.getElementById('diagram').innerHTML = markup; }, svg);
    const element = page.locator('#diagram svg');
    const bounds = await element.boundingBox();
    if (!bounds || bounds.width < 100 || bounds.height < 100) throw new Error(`Unexpected rendered size for ${name}: ${JSON.stringify(bounds)}`);
    const stem = name.replace(/\.mmd$/, '');
    fs.writeFileSync(path.join(outputDir, `${stem}.svg`), svg, 'utf8');
    await element.screenshot({ path: path.join(outputDir, `${stem}.png`), timeout: 30000 });
    console.log(JSON.stringify({ name, width: Math.round(bounds.width), height: Math.round(bounds.height), deviceScaleFactor: 3 }));
  }
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });
