from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlsplit,unquote
import re,subprocess,json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'docs';errors=[]
parsed={p:BeautifulSoup(p.read_text(),'html.parser') for p in D.rglob('*.html')}
links=0
for p,s in parsed.items():
 ids=[t['id'] for t in s.select('[id]')]
 if len(ids)!=len(set(ids)):errors.append(f'Duplicate IDs {p.name}')
 if len(s.select('h1'))!=1:errors.append(f'H1 count {p.name}')
 if any(not x.get('alt') for x in s.select('img')):errors.append(f'Image alt {p.name}')
 for t in s.select('[href],[src]'):
  href=t.get('href',t.get('src'));u=urlsplit(href)
  if u.scheme or u.netloc:continue
  dest=(p.parent/unquote(u.path)).resolve() if u.path else p
  if dest.is_dir():dest=dest/'index.html'
  if not dest.exists():errors.append(f'Missing {p.name}: {href}');continue
  if u.fragment and dest.suffix=='.html':
   if not parsed[dest].select('[id="'+u.fragment+'"]'):errors.append(f'Anchor {p.name}: {href}')
  links+=1
 for field in ['og:title','og:description','og:image','og:url']:
  if not s.find('meta',property=field):errors.append(f'Missing {field} {p.name}')
 if s.find('meta',attrs={'name':'twitter:card'}).get('content')!='summary_large_image':errors.append(f'Social card {p.name}')
 # Known private implementation markers, while preserving public source code in the book.
 if any(x in p.read_text() for x in ['/workspace/scratch/','ghp_','github_pat_','sk-proj-']):errors.append(f'Private marker {p.name}')
raw=(ROOT/'source/manuscript.md').read_text();chunks=list(re.finditer(r'<a id="([^"]+)"></a>\n# ([^\n]+)',raw))
for i,m in enumerate(chunks):
 text=raw[m.start():chunks[i+1].start() if i+1<len(chunks) else len(raw)]
 number=re.match(r'(\d+)\s+',m[2]);slug=f'{int(number[1]):02}.html' if number else {'glossary':'glossary.html','works-cited':'references.html'}.get(m[1],'start.html')
 expected=BeautifulSoup(subprocess.run(['pandoc','--from=gfm','--to=html5','--no-highlight'],input=text,text=True,capture_output=True,check=True).stdout,'html.parser')
 actual=BeautifulSoup(str(parsed[D/'chapters'/slug].select_one('article')),'html.parser')
 for element in actual.select('.glossary-filter,#glossary-count'):element.decompose()
 norm=lambda s:re.sub(r'\s+',' ',s.get_text(' ',strip=True)).strip()
 if norm(expected)!=norm(actual):errors.append('Text parity '+slug)
 expected_codes=[t.get_text() for t in expected.select('pre code')];actual_codes=[t.get_text() for t in actual.select('pre code')]
 if expected_codes!=actual_codes:errors.append('Code parity '+slug)
expected_downloads=json.loads((ROOT/'source/download-manifest.json').read_text())
for name, expected in expected_downloads.items():
 data=(D/'downloads'/name).read_bytes()
 if hashlib.sha256(data).hexdigest()!=expected['sha256'] or len(data)!=expected['bytes']:errors.append('Download parity '+name)
if (ROOT/'source/manuscript.md').read_bytes()!=(D/'downloads/training-your-own-models.md').read_bytes():errors.append('Manuscript download mismatch')
with zipfile.ZipFile(D/'downloads/training-models-companion.zip') as z:
 if z.testzip():errors.append('ZIP CRC')
 zipnames=z.namelist()
 script_count=len([n for n in zipnames if n.endswith('.py')]);assert script_count==57,script_count
s=parsed[D/'chapters/glossary.html'];assert len(s.select('article>ul>li'))==243
search=json.loads((D/'assets/search-index.js').read_text().removeprefix('window.BOOK_SEARCH=').removesuffix(';\n'))
for hit in search:
 u=urlsplit(hit['url']);p=D/u.path
 if not parsed[p].select('[id="'+u.fragment+'"]'):errors.append('Search anchor '+hit['url'])
result={'html_pages':len(parsed),'manuscript_sections':len(chunks),'glossary_terms':243,'scripts':script_count,'search_sections':len(search),'internal_links_assets_checked':links,'errors':errors}
print(json.dumps(result,indent=2));(ROOT/'qa/validation.json').write_text(json.dumps(result,indent=2)+'\n');assert not errors
