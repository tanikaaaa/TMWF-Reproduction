"""Plot verified archived standalone results; no model or dataset loading.

Requires Python standard library and macOS AppKit via osascript (no pip packages).
Run with --check to validate only; default creates exactly two PNGs, refusing
existing outputs. Layout/fonts/colors are fixed; raster bytes can vary by macOS.
"""
import argparse
import base64
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'docs/TMWF_REPRODUCTION_REPORT.md'
METRICS = ('Accuracy', 'Precision', 'Recall')
W, H, DPI = 2700, 2100, 300
BLUE, ORANGE, INK, MUTED = '#2876A3', '#D17A32', '#202B33', '#52616B'
EXPECTED_MARKERS = {
 'appendix_f_table7': 'PASS: exactly 5000 samples per dataset; checkpoint and model state unchanged. No files written.',
 'appendix_h_ratio_real_table9': 'PASS: 6000 component predictions; 3000 samples per mixed set; model parameters/buffers, checkpoint and source dataset hashes unchanged.',
 'appendix_h_delay_table9': 'PASS: 3000 predictions; checkpoint, model parameters/buffers, datasets and generation sources unchanged.',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence():
    candidates = []
    for path in (ROOT / 'outputs/reproducibility').glob('*/summary.json'):
        s = json.loads(path.read_text())
        if s.get('execution_status') == 'success':
            candidates.append((datetime.fromisoformat(s['end_timestamp']), path, s))
    require(bool(candidates), 'No successful archive')
    _, summary_path, summary = max(candidates, key=lambda x: (x[0], str(x[1])))
    require(summary.get('integrity_unchanged') is True and
            summary['hashes_before'] == summary['hashes_after'], 'Archive integrity failed')
    require(len(summary['evaluators']) == 3 and
            {e['evaluator'] for e in summary['evaluators']} == set(EXPECTED_MARKERS),
            'Expected exactly three named evaluators')
    rows, protected = {}, {summary_path: sha(summary_path), REPORT: sha(REPORT)}
    def add(job, metric, count, reproduced, published):
        key = (job, metric)
        require(key not in rows, f'Duplicate metric: {key}')
        pub, rep = Decimal(published), Decimal(reproduced)
        require(0 <= pub <= 100 and 0 <= rep <= 100, 'Percentage outside 0–100')
        rows[key] = (int(count), pub, rep)
    for e in summary['evaluators']:
        # Resolve by basename within the selected archive, portable across checkouts.
        path = summary_path.parent / Path(e['log_path']).name
        require(sha(path) == e['log_sha256'], f'Log hash mismatch: {path}')
        protected[path] = sha(path)
        log = path.read_text()
        require(e['exit_code'] == 0 and e['execution_status'] == 'success', 'Evaluator failed')
        require(EXPECTED_MARKERS[e['evaluator']] in log and
                all(m in log for m in e['expected_markers']), 'Missing completion marker')
        if e['evaluator'] == 'appendix_h_delay_table9':
            require('NOT literal released-code execution' in log, 'Missing delay disclosure')
            matches = re.findall(r'^(Basic (?:Accuracy|Precision|Recall))\s+([\d.]+)\s+([\d.]+)\s+[+-][\d.]+$', log, re.M)
            require(len(matches) == 3, 'Expected three delay metrics')
            for metric, rep, pub in matches:
                add('mixed_delay', metric, 3000, rep, pub)
        else:
            matches = re.findall(r'^(test_[246]tab|mixed_ratio|mixed_real|real_[234]tab)\s+(\d+)\s+((?:Basic|Advanced) (?:Accuracy|Precision|Recall))\s+([\d.]+)\s+([\d.]+)\s+[+-][\d.]+$', log, re.M)
            require(len(matches) == (18 if e['evaluator'] == 'appendix_f_table7' else 15), 'Incorrect metric count')
            for job, count, metric, rep, pub in matches:
                add(job, metric, count, rep, pub)
    expected = {(f'test_{n}tab', f'{kind} {m}') for n in (2,4,6)
                for kind in ('Basic','Advanced') for m in METRICS}
    expected |= {(j, f'Basic {m}') for j in ('mixed_ratio','mixed_delay','mixed_real',
                 'real_2tab','real_3tab','real_4tab') for m in METRICS}
    require(set(rows) == expected and len(rows) == 36, 'Expected all 36 primary metrics')
    for (job, metric), (count, pub, rep) in rows.items():
        require(count == (5000 if job.startswith('test_') else 3000 if job.startswith('mixed_') else 1000), 'Incorrect sample count')
    report_rows = {}
    for line in REPORT.read_text().splitlines():
        c = [x.strip() for x in line.split('|')]
        if len(c) != 7 or not re.fullmatch(r'[+-]\d+\.\d', c[5]):
            continue
        job = c[1].split()[0]
        metric = c[2].removeprefix('Overall ')
        if not metric.startswith(('Basic','Advanced')):
            metric = 'Basic ' + metric
        key = (job, metric)
        require(key not in report_rows, 'Duplicate report metric')
        pub, rep, delta = map(Decimal, c[3:6])
        require(rep - pub == delta, f'Report difference mismatch: {key}')
        count = int(re.search(r'\(([\d,]+)\)', c[1])[1].replace(',', ''))
        report_rows[key] = (count, pub, rep)
    require(report_rows == rows, 'Report/archive mismatch')
    print(f'PASS evidence: {summary_path.relative_to(ROOT)}; 3 successful evaluators, log hashes, 36 metrics and report agree.')
    return rows, protected, summary_path.parent.name


def drawing(rows, table):
    commands = []
    def text(value, x, y, size=32, align='left', color=INK, bold=False):
        commands.append(dict(kind='text', text=value, x=x, y=y, size=size, align=align, color=color, bold=bold))
    def rect(x,y,w,h,color):
        commands.append(dict(kind='rect',x=x,y=y,w=w,h=h,color=color))
    def line(x,y,x2,y2,color='#DDE2E6',width=2):
        commands.append(dict(kind='line',x=x,y=y,x2=x2,y2=y2,color=color,width=width))
    text(f'TMWF · Appendix {"F" if table == 7 else "H"} / Table {table}',110,65,54,bold=True)
    text('Synthetic open-world evaluation' if table == 7 else 'Synthetic and real open-world evaluation · Overall Basic metrics',110,137,35,color=MUTED)
    rect(110,211,45,29,BLUE);text('Published',172,205,33)
    rect(430,211,45,29,ORANGE);text('Reproduced · single run, epoch 50',492,205,33)
    text('Metric value (%) · identical 0–100 scale in every panel',110,281,30,color=MUTED)
    if table == 7:
        panels=[(f'test_{n}tab',kind,f'{n}-tab · Overall {kind}', 'n = 5,000') for kind in ('Basic','Advanced') for n in (2,4,6)]
    else:
        panels=[(j,'Basic',title,sub) for j,title,sub in (
            ('mixed_ratio','Mixed ratio-synthetic','n = 3,000'),
            ('mixed_delay','Mixed delay-synthetic','n = 3,000 · copy-isolated'),
            ('mixed_real','Mixed real-world','n = 3,000'),
            ('real_2tab','Real 2-tab','n = 1,000'),
            ('real_3tab','Real 3-tab','n = 1,000'),
            ('real_4tab','Real 4-tab','n = 1,000'))]
    plotted = []
    for i,(job,kind,title,sub) in enumerate(panels):
        col,row=i%3,i//3
        left=150+col*850; top=365+row*755; bottom=top+570; width=680; height=440
        text(title,left,top,36,bold=True)
        text(sub,left,top+49,29,color=MUTED)
        for tick in range(0,101,20):
            y=bottom-height*tick/100
            line(left,y,left+width,y)
            text(str(tick),left-19,y-17,27,'right',MUTED)
        line(left,bottom,left+width,bottom,INK)
        for m,metric in enumerate(METRICS):
            center=left+113+m*227
            count,pub,rep=rows[job,f'{kind} {metric}']
            for k,(value,color) in enumerate(((pub,BLUE),(rep,ORANGE))):
                x=center-78+k*82; bar_h=height*float(value)/100
                rect(x,bottom-bar_h,70,bar_h,color)
                text(f'{value:.1f}',x+35,bottom-bar_h-43,29,'center')
            text(metric,center,bottom+26,30,'center')
            plotted.append((job,f'{kind} {metric}',pub,rep))
        if job=='mixed_delay':
            delta=rows[job,'Basic Accuracy'][2]-rows[job,'Basic Accuracy'][1]
            text(f'Accuracy difference: {delta:+.1f} pp',left,top+95,28,color=MUTED)
    require(len(plotted)==18 and len({p[:2] for p in plotted})==18,'Incorrect plotted metric count')
    text('Published values: TMWF paper. Reproduced values: archived standalone evaluation.',110,1910,29,color=MUTED)
    text('Single reproduction run; no confidence intervals or statistical significance claims.',110,1954,29,color=MUTED)
    text('Delay: copy-isolated released algorithm, not literal released-code execution.' if table==9
         else 'Basic metrics ignore label order; Advanced metrics retain prediction-slot order.',110,1998,29,color=MUTED)
    return commands


# Native macOS raster graphics, in memory. Text extents are checked before output.
JXA = r'''
ObjC.import('AppKit');
const W=2700,H=2100;
const bitmap=$.NSBitmapImageRep.alloc.initWithBitmapDataPlanesPixelsWidePixelsHighBitsPerSampleSamplesPerPixelHasAlphaIsPlanarColorSpaceNameBytesPerRowBitsPerPixel(null,W,H,8,4,true,false,$.NSDeviceRGBColorSpace,0,0);
const context=$.NSGraphicsContext.graphicsContextWithBitmapImageRep(bitmap);
$.NSGraphicsContext.saveGraphicsState;
$.NSGraphicsContext.setCurrentContext(context);
function color(hex){return $.NSColor.colorWithCalibratedRedGreenBlueAlpha(parseInt(hex.slice(1,3),16)/255,parseInt(hex.slice(3,5),16)/255,parseInt(hex.slice(5,7),16)/255,1);}
function rect(x,y,w,h){return $.NSMakeRect(x,H-y-h,w,h);}
color('#FFFFFF').setFill;
$.NSBezierPath.bezierPathWithRect(rect(0,0,W,H)).fill;
for(const c of COMMANDS){
 if(c.kind==='rect'){color(c.color).setFill;$.NSBezierPath.bezierPathWithRect(rect(c.x,c.y,c.w,c.h)).fill;}
 else if(c.kind==='line'){color(c.color).setStroke;const p=$.NSBezierPath.bezierPath;p.lineWidth=c.width;p.moveToPoint($.NSMakePoint(c.x,H-c.y));p.lineToPoint($.NSMakePoint(c.x2,H-c.y2));p.stroke;}
 else {
  const font=$.NSFont.fontWithNameSize(c.bold?'Helvetica-Bold':'Helvetica',c.size);
  const attrs=$.NSMutableDictionary.alloc.init;
  attrs.setObjectForKey(font,$.NSFontAttributeName);attrs.setObjectForKey(color(c.color),$.NSForegroundColorAttributeName);
  const s=$(c.text),size=s.sizeWithAttributes(attrs);let x=c.x;
  if(c.align==='center')x-=size.width/2;else if(c.align==='right')x-=size.width;
  if(x<0||x+size.width>W||c.y<0||c.y+size.height>H)throw Error('Clipped label: '+c.text);
  s.drawAtPointWithAttributes($.NSMakePoint(x,H-c.y-size.height),attrs);
 }
}
$.NSGraphicsContext.restoreGraphicsState;
ObjC.unwrap(bitmap.representationUsingTypeProperties($.NSBitmapImageFileTypePNG,$.NSDictionary.dictionary).base64EncodedStringWithOptions(0));
'''


def png_chunk(kind, data):
    return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)


def render(commands):
    require(sys.platform=='darwin','Native renderer requires macOS AppKit and osascript')
    source='const COMMANDS='+json.dumps(commands,ensure_ascii=True)+';\n'+JXA
    result=subprocess.run(['/usr/bin/osascript','-l','JavaScript','-'],input=source,text=True,capture_output=True,check=True)
    raw=base64.b64decode(result.stdout.strip(),validate=True)
    require(raw[:8]==b'\x89PNG\r\n\x1a\n','Renderer did not return PNG')
    # Normalize resolution metadata; omit variable timestamp metadata.
    output=raw[:8];offset=8
    while offset<len(raw):
        size=struct.unpack('>I',raw[offset:offset+4])[0]
        kind=raw[offset+4:offset+8];data=raw[offset+8:offset+8+size]
        require(zlib.crc32(kind+data)&0xffffffff==struct.unpack('>I',raw[offset+8+size:offset+12+size])[0],'PNG CRC mismatch')
        if kind not in (b'pHYs',b'tIME'):
            output+=png_chunk(kind,data)
        if kind==b'IHDR':
            require(struct.unpack('>II',data[:8])==(W,H),'Incorrect image dimensions')
            ppm=round(DPI/0.0254);output+=png_chunk(b'pHYs',struct.pack('>IIB',ppm,ppm,1))
        offset+=12+size
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Verify evidence without rendering or writing files')
    args=parser.parse_args()
    rows,protected,archive=evidence()
    if args.check:
        return
    paths=[ROOT/f'docs/figures/tmwf_table{n}_comparison.png' for n in (7,9)]
    for path in paths:
        require(not path.exists() and not path.is_symlink(),f'Refusing to overwrite {path}')
        require(not path.parent.is_symlink(),'Symlink figure directory refused')
    images=[render(drawing(rows,n)) for n in (7,9)]
    require(all(sha(p)==h for p,h in protected.items()),'Evidence changed during rendering')
    paths[0].parent.mkdir(parents=True,exist_ok=True)
    for path,png in zip(paths,images):
        with path.open('xb') as handle:
            handle.write(png)
        print(f'CREATED {path.relative_to(ROOT)}: {W} x {H}, {DPI} dpi, {len(png):,} bytes; SHA-256 {sha(path)}')
    print('PASS: 18 metrics per figure; 72 bars total; native text bounds checked; no source artifacts changed.')
    print('Captions: Table 7 — Published and single-run reproduced Overall Basic/Advanced results on 5,000 traces per tab count.')
    print('Table 9 — Published and single-run reproduced Overall Basic results; mixed groups contain 3,000 traces, individual real groups 1,000. Delay uses copy-isolated synthesis.')


if __name__=='__main__':
    main()
