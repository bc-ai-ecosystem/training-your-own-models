#!/usr/bin/env python3
"""Build the complete static web edition. Requires Python 3, BeautifulSoup4 and Pandoc."""
from pathlib import Path
import re, subprocess, html, json, shutil, hashlib
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parent; DOCS=ROOT/'docs'; SRC=ROOT/'source/manuscript.md'
PUBLIC='https://mrscripty.github.io/training-your-own-models/'
TITLE='Training Your Own Models on One 24 GB GPU'
DESC='A project-first handbook by Dr.Puma: 40 chapters, 243 glossary terms, interactive learning tools, and a 57-script companion. Start small. Understand every training step.'
def esc(s): return html.escape(str(s),quote=True)
raw=SRC.read_text(); starts=list(re.finditer(r'<a id="([^"]+)"></a>\n# ([^\n]+)',raw)); pages=[]; targets={}
for i,m in enumerate(starts):
 text=raw[m.start():starts[i+1].start() if i+1<len(starts) else len(raw)]
 title=m[2];num=re.match(r'(\d+)\s+',title)
 slug=f'{int(num[1]):02}.html' if num else {'glossary':'glossary.html','works-cited':'references.html'}.get(m[1],'start.html')
 p={'slug':slug,'title':title,'raw':text,'anchor':m[1],'n':int(num[1]) if num else None};pages.append(p)
 for anchor in re.findall(r'<a id="([^"]+)"></a>',text):
  assert anchor not in targets, anchor
  targets[anchor]='chapters/'+slug
assert len([p for p in pages if p['n']])==40
manifest={'chapters':40,'glossary_terms':len(re.findall(r'^- \*\*',next(p['raw'] for p in pages if p['slug']=='glossary.html'),re.M)),'anchors':len(targets),'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest()}
assert manifest['glossary_terms']==243

def nav(base='',current=None):
 return '<nav aria-label="Book contents"><label class="nav-filter">Find a chapter<input type="search" class="chapter-filter" placeholder="Filter titles" aria-label="Filter chapter titles"></label>'+''.join(f'<a class="chapter-link" href="{base}chapters/{p["slug"]}"'+(' aria-current="page"' if p['slug']==current else '')+f'><span>{p["n"] or ("A–Z" if p["slug"]=="glossary.html" else "Ref" if p["slug"]=="references.html" else "→")}</span>{esc(re.sub(r"^\d+\s+","",p["title"]))}</a>' for p in pages)+'</nav>'
def shell(title,body,rel,base=''):
 url=PUBLIC+('' if rel=='index.html' else rel);full=TITLE if rel=='index.html' else title+' · Training Your Own Models'
 tags=f'<meta name="description" content="{esc(DESC)}"><link rel="canonical" href="{url}"><meta property="og:type" content="website"><meta property="og:site_name" content="Training Your Own Models"><meta property="og:title" content="{esc(full)}"><meta property="og:description" content="{esc(DESC)}"><meta property="og:url" content="{url}"><meta property="og:image" content="{PUBLIC}assets/social-preview.jpg"><meta property="og:image:secure_url" content="{PUBLIC}assets/social-preview.jpg"><meta property="og:image:type" content="image/jpeg"><meta property="og:image:width" content="1280"><meta property="og:image:height" content="640"><meta property="og:image:alt" content="Training Your Own Models. On One 24 GB GPU. Vibe Authored by Dr.Puma. Cream, blue and orange scientific cover artwork."><meta name="twitter:card" content="summary_large_image"><meta name="twitter:title" content="{esc(full)}"><meta name="twitter:description" content="{esc(DESC)}"><meta name="twitter:image" content="{PUBLIC}assets/social-preview.jpg"><meta name="twitter:image:alt" content="Training Your Own Models, Vibe Authored by Dr.Puma">'
 return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#f6f3eb"><title>{esc(full)}</title>{tags}<link rel="icon" href="{base}assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="{base}assets/style.css"></head><body><a class="skip" href="#main">Skip to content</a><header class="topbar"><a class="brand" href="{base}index.html"><span class="brand-mark" aria-hidden="true">→</span><span>Training Your<br>Own Models</span></a><nav aria-label="Main navigation"><a href="{base}chapters/start.html">Read the book</a><a href="{base}labs.html">Explore</a><a href="{base}search.html" aria-label="Search the book">Search</a><a href="{base}downloads.html">Downloads</a></nav></header>{body}<footer class="footer"><div><strong>Training Your Own Models</strong><br>Vibe Authored by Dr.Puma · 2 October 2026</div><div><a href="{base}chapters/glossary.html">Glossary</a><a href="{base}chapters/references.html">References</a><a href="https://github.com/MrScripty/training-your-own-models">GitHub →</a></div></footer><script src="{base}assets/site.js" defer></script></body></html>'''

search=[]
for i,p in enumerate(pages):
 rendered=subprocess.run(['pandoc','--from=gfm','--to=html5','--no-highlight'],input=p['raw'],capture_output=True,text=True,check=True).stdout
 soup=BeautifulSoup(rendered,'html.parser')
 # The manuscript's explicit stable anchors are authoritative; remove Pandoc auto IDs to avoid collisions.
 for a in soup.find_all('a',id=True):
  if a.parent.name=='p' and not a.parent.get_text(strip=True): a.parent.unwrap()
 for heading in soup.find_all(re.compile('^h[1-6]$')):
  heading.attrs.pop('id',None)
 for a in soup.find_all('a',href=True):
  if a['href'].startswith('#'):
   anchor=a['href'][1:];assert anchor in targets,anchor
   a['href']=(('#' if targets[anchor]=='chapters/'+p['slug'] else '../'+targets[anchor]+'#')+anchor)
 for img in soup.find_all('img'):
  assert img['src'].startswith('assets/'),img['src']
  img['src']='../'+img['src'];img['loading']='lazy'
 for table in soup.find_all('table'):
  wrap=soup.new_tag('div',attrs={'class':'table-scroll','tabindex':'0','role':'region','aria-label':'Scrollable table'});table.wrap(wrap)
 toc=[]
 for a in soup.find_all('a',id=True):
  h=a.find_next_sibling()
  if h and h.name=='h2':toc.append((a['id'],h.get_text(' ',strip=True)))
 # Search keeps all manuscript content, grouped by original headings.
 for h in soup.find_all(re.compile('^h[12]$')):
  prev=h.find_previous_sibling('a');anchor=prev.get('id') if prev else p['anchor'];bits=[h.get_text(' ',strip=True)]
  for el in h.next_siblings:
   if getattr(el,'name',None) in ['h1','h2']:break
   bits.append(el.get_text(' ',strip=True) if hasattr(el,'get_text') else str(el))
  search.append({'title':h.get_text(' ',strip=True),'chapter':p['title'],'url':'chapters/'+p['slug']+'#'+anchor,'text':' '.join(bits)})
 extra=''
 if p['n'] in [3,12,15]:
  lab={3:('gradient','Watch a parameter learn'),12:('attention','Inspect a causal attention mask'),15:('memory','Calculate persistent training-state memory')}[p['n']]
  extra=f'<aside class="callout"><span class="eyebrow">Interactive companion</span><h2>{lab[1]}</h2><p>Change one assumption in a small, transparent browser example.</p><a href="../labs.html#{lab[0]}">Open the learning tool →</a></aside>'
 if p['slug']=='glossary.html':
  extra='<label class="glossary-filter">Find a term<input id="glossary-filter" type="search" placeholder="Try gradient, LoRA, or tensor"></label><p id="glossary-count" class="muted" aria-live="polite">243 terms</p>'
  first_h=soup.find('h1');first_h.insert_after(BeautifulSoup(extra,'html.parser'));extra=''
 prev=pages[i-1] if i else None;nxt=pages[i+1] if i+1<len(pages) else None
 pager='<nav class="pager" aria-label="Adjacent chapters">'+(f'<a href="{prev["slug"]}"><small>← Previous</small>{esc(prev["title"])}</a>' if prev else '<a href="../index.html"><small>← Home</small>Back to the cover</a>')+(f'<a href="{nxt["slug"]}"><small>Next →</small>{esc(nxt["title"])}</a>' if nxt else '<a href="../downloads.html"><small>Keep learning →</small>Get the companion</a>')+'</nav>'
 body=f'<div class="reader"><aside class="sidebar">{nav("../",p["slug"])}</aside><main class="prose" id="main"><details class="mobile-toc"><summary>Browse the book</summary>{nav("../",p["slug"])}</details><div class="chapter-meta"><span class="eyebrow">'+('Chapter '+str(p['n'])+' / 40' if p['n'] else 'Reading guide' if p['slug']=='start.html' else 'Reference')+f'</span><span>{max(1,round(len(p["raw"].split())/220))} min read</span></div><article class="manuscript">{soup}</article>{extra}{pager}</main><aside class="on-page"><h2>On this page</h2>'+''.join(f'<a href="#{a}">{esc(t)}</a>' for a,t in toc)+'</aside></div>'
 (DOCS/'chapters'/p['slug']).write_text(shell(p['title'],body,'chapters/'+p['slug'],'../'))

cards=''.join(f'<a class="chapter-card" href="chapters/{p["slug"]}"><span class="number">{p["n"]:02}</span><h3>{esc(re.sub(r"^\d+\s+","",p["title"]))}</h3><span aria-hidden="true">→</span></a>' for p in pages if p['n'])
body=f'''<main id="main"><section class="hero wrap"><div><p class="eyebrow">A project-first handbook · Web edition</p><h1>Training<br>Your Own<br><em>Models.</em></h1><p class="hero-subtitle">On One 24 GB GPU</p><p class="lede">Start with three parameters. Learn what changes, why it changes, and how to know whether it worked.</p><p class="author">Vibe Authored by Dr.Puma</p><div class="actions"><a class="button" href="chapters/start.html">Start reading <span>→</span></a><a class="text-link" href="labs.html">Try a learning tool →</a></div></div><figure class="cover"><img src="assets/cover.png" width="1104" height="1424" alt="Training Your Own Models book cover with orange data becoming blue model surfaces. Vibe Authored by Dr.Puma."></figure></section><section class="stats wrap" aria-label="Inside the book"><div><strong>40</strong><span>practical chapters</span></div><div><strong>243</strong><span>linked glossary terms</span></div><div><strong>57</strong><span>companion scripts</span></div><div><strong>247</strong><span>PDF pages, including covers</span></div></section><section class="section wrap"><div class="section-heading"><div><p class="eyebrow">Small experiments. Clear thinking.</p><h2>Build the understanding<br>before the bigger model.</h2></div><p>From your first classifier to language, vision, speech, and sound. Follow the complete book, inspect a definition, or explore one idea in your browser.</p></div><div class="feature-grid"><a class="feature" href="labs.html#gradient"><span>01 / Explore</span><h3>Watch a parameter learn</h3><p>Take gradient steps. Change the learning rate. See the loss respond.</p><b>Open the experiment →</b></a><a class="feature" href="labs.html#memory"><span>02 / Plan</span><h3>Make the memory visible</h3><p>Separate weights, gradients, and optimizer states before planning a run.</p><b>Try the calculator →</b></a><a class="feature" href="chapters/glossary.html"><span>03 / Connect</span><h3>Find the missing definition</h3><p>Every glossary term links back to an explanation in the book.</p><b>Browse the glossary →</b></a></div></section><section class="section contents wrap" id="contents"><div class="section-heading"><div><p class="eyebrow">The complete book</p><h2>One project at a time.</h2></div><p>New to neural networks? Begin with <a href="chapters/start.html">the reading guide</a>, then work through the chapters in order.</p></div><div class="chapter-grid">{cards}</div><div class="reference-row"><a href="chapters/glossary.html">Glossary A–Z →</a><a href="chapters/references.html">Works cited →</a><a href="search.html">Search the full book →</a></div></section><section class="evidence wrap"><div><p class="eyebrow">Read the evidence carefully</p><h2>A recipe is a starting point.</h2></div><div><p>No GPU training or 24 GB peak-memory measurements were performed for this edition. The book distinguishes documented capabilities, worked calculations, proposed configurations, and explicitly recorded checks.</p><p><a href="chapters/40.html">Read the verification record →</a></p></div></section></main>'''
(DOCS/'index.html').write_text(shell(TITLE,body,'index.html'))
files=[('training-your-own-models.pdf','Complete PDF','247 pages, including the front and back covers.'),('training-your-own-models.md','Editable Markdown','The complete manuscript, unchanged. Its five figures are in the companion archive.'),('training-models-companion.zip','Companion archive','57 scripts, source manuscript, figures, configurations, and the companion README.')]
checks={}
for name,_,_ in files:checks[name]={'bytes':(DOCS/'downloads'/name).stat().st_size,'sha256':hashlib.sha256((DOCS/'downloads'/name).read_bytes()).hexdigest()}
body='<main class="narrow section" id="main"><p class="eyebrow">Keep a copy</p><h1>Read. Inspect. Reproduce.</h1><p class="lede">The same book, in the format that suits your work. The companion files are the authoritative copy for running the examples.</p><div class="download-list">'+''.join(f'<a href="downloads/{f}" download><div><h2>{t} ↓</h2><p>{d}</p></div><span>{checks[f]["bytes"]/1024/1024:.1f} MB</span></a>' for f,t,d in files)+'</div><div class="callout"><h2>Before running a recipe</h2><p>Read the companion README and the <a href="chapters/40.html">verification record</a>. Use separate environments for different project families. GPU recipes are proposed experiments, not measured results from your card.</p></div><details><summary>Verify download integrity</summary><p>SHA-256 hashes of the exact published files:</p>'+''.join(f'<h3>{f}</h3><code class="hash">{v["sha256"]}</code>' for f,v in checks.items())+'</details><figure class="back-cover"><img src="assets/back-cover.png" alt="Back cover of Training Your Own Models" loading="lazy"></figure></main>'
(DOCS/'downloads.html').write_text(shell('Downloads',body,'downloads.html'))
(DOCS/'downloads/checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
body='<main class="narrow section" id="main"><p class="eyebrow">Across all 40 chapters</p><h1>Find an idea.</h1><label class="search-label" for="book-search">Search the full book</label><input id="book-search" type="search" placeholder="Try loss masking, embeddings, or memory" autocomplete="off"><p class="muted">Search matches all the words you enter. Results link to their original section. Everything runs in your browser.</p><p id="search-status" aria-live="polite"></p><div id="search-results"></div></main><script src="assets/search-index.js"></script>'
(DOCS/'search.html').write_text(shell('Search the book',body,'search.html'))
(DOCS/'assets/search-index.js').write_text('window.BOOK_SEARCH='+json.dumps(search,ensure_ascii=False)+';\n')
body='<main class="narrow section" id="main"><h1>This page is not in the book.</h1><p><a href="'+PUBLIC+'">Return to the cover</a> or <a href="'+PUBLIC+'search.html">search the book</a>.</p></main>'
(DOCS/'404.html').write_text(shell('Page not found',body,'404.html',PUBLIC))
(DOCS/'.nojekyll').touch()
(DOCS/'assets/favicon.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><rect width="48" height="48" rx="10" fill="#144cac"/><path d="M12 34 33 13M17 13h16v16" stroke="#fff9ea" stroke-width="5" fill="none"/></svg>')
from labs import BODY
(DOCS/'labs.html').write_text(shell('Interactive learning tools',BODY,'labs.html'))
(DOCS/'sitemap.xml').write_text('<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'+''.join('<url><loc>'+PUBLIC+p.relative_to(DOCS).as_posix()+'</loc></url>' for p in sorted(DOCS.rglob('*.html')) if p.name!='404.html')+'</urlset>')
(ROOT/'qa/content-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
# Labs are a supplement, not a replacement for manuscript text.
print(json.dumps(manifest))
