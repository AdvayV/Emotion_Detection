"""Use the packaged DOCX rasterizer with Word export on Windows."""
import importlib.util,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
runtime=Path(os.environ.get('CODEMOOD_DOCUMENT_RUNTIME',Path.home()/'.cache/codex-runtimes/codex-primary-runtime'))
helper=runtime/'plugins/openai-primary-runtime/plugins/documents/skills/documents/render_docx.py'
poppler=runtime/'dependencies/native/poppler/Library/bin'
os.environ['PATH']=str(poppler)+os.pathsep+os.environ['PATH']
os.environ['TEMP']=str(root/'.agent-state/test-temp');os.environ['TMP']=os.environ['TEMP']
spec=importlib.util.spec_from_file_location('packaged_render_docx',helper)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def word_pdf(doc_path,user_profile,convert_tmp_dir,stem,verbose=False):
 output=str(Path(convert_tmp_dir)/(stem+'.pdf'))
 subprocess.run([str(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'),'-NoProfile','-File',str(root/'scripts/export_submission_pdf.ps1'),'-InputPath',doc_path,'-OutputPath',output],check=True,timeout=120)
 return output,'Word native PDF export'
module.convert_to_pdf=word_pdf
if __name__=='__main__':module.main()
