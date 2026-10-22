/** Render editable Mermaid sources to SVG plus print-quality PNG fallback. */
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.argv[2]);
const mermaidScript = process.argv[3];
const root = path.resolve(__dirname,'..');
(async()=>{
 const edge='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
 const executablePath=process.env.BROWSER_EXECUTABLE || (fs.existsSync(edge)?edge:undefined);
 const browser=await chromium.launch({headless:true,...(executablePath?{executablePath}:{})});
 const page=await browser.newPage({viewport:{width:1600,height:1800},deviceScaleFactor:3});
 await page.setContent('<!doctype html><html><body style="margin:0;background:white"><div id="diagram"></div></body></html>');
 await page.addScriptTag({path:mermaidScript});
 await page.evaluate(()=>mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'base',fontFamily:'Arial',themeVariables:{fontSize:'20px',fontFamily:'Arial',primaryColor:'#edf6f5',primaryTextColor:'#12233f',primaryBorderColor:'#168c88',lineColor:'#64748b',clusterBkg:'#f8fafc',clusterBorder:'#cbd5e1',tertiaryColor:'#f1f5f9'},flowchart:{htmlLabels:false,nodeSpacing:28,rankSpacing:36,padding:18,curve:'basis',useMaxWidth:false}}));
 const out=path.join(root,'docs/submission/figures');fs.mkdirSync(out,{recursive:true});
 for(const name of fs.readdirSync(path.join(root,'docs/submission/diagrams')).filter(x=>x.endsWith('.mmd'))){
  const code=fs.readFileSync(path.join(root,'docs/submission/diagrams',name),'utf8');
  const svg=await page.evaluate(async(code)=>{const r=await mermaid.render('flow',code);return r.svg;},code);
  await page.evaluate(svg=>{document.getElementById('diagram').innerHTML=svg;},svg);
  const element=page.locator('#diagram svg');const rect=await element.boundingBox();
  fs.writeFileSync(path.join(out,name.replace('.mmd','.svg')),svg);
  await element.screenshot({path:path.join(out,name.replace('.mmd','.png')),timeout:30000});
  console.log(JSON.stringify({name,width:Math.round(rect.width),height:Math.round(rect.height),scale:3}));
 }
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
