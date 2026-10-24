"""Draw source-grounded TMWF diagrams with vector primitives, exported as PNG.

Uses the existing plot_tmwf_results native macOS renderer without modifying it.
No model imports, forward passes, dataset reads, or training. --check is read-only.
Output files are exclusively created, never overwritten. B denotes batch size.
"""
import argparse
import ast
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from plot_tmwf_results import render, sha, require, BLUE, ORANGE, INK, MUTED

ROOT = Path(__file__).resolve().parents[1]
MODEL = dict(embed_dim=256,nhead=8,dim_feedforward=1024,num_encoder_layers=2,
             num_decoder_layers=2,max_len=121,num_queries=6,cls=51,dropout=0.1)


def assigned(path, name):
    tree=ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            value=node.value
            if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='dict':
                return {k.arg:ast.literal_eval(k.value) for k in value.keywords}
            return ast.literal_eval(value)
    raise ValueError(f'Missing {name} in {path}')


def validate():
    for family in ('f','h'):
        require(assigned(ROOT/f'code/train_appendix_{family}.py','MODEL')==MODEL,'Model settings changed')
    f=assigned(ROOT/'code/run_appendix_f_synthesis.py','JOBS')
    require(f=={'train_6tab':('train',6,20000),'test_2tab':('test',2,5000),
                'test_4tab':('test',4,5000),'test_6tab':('test',6,5000)},'F jobs changed')
    h=assigned(ROOT/'code/run_appendix_h_synthesis.py','JOBS')
    require(h=={'train_6tab':('train',6,10,40000),'ratio_test_2tab':('test',2,1,1000),
                'ratio_test_3tab':('test',3,1,1000),'ratio_test_4tab':('test',4,1,1000)},'H jobs changed')
    require(assigned(ROOT/'code/run_appendix_h_delay_synthesis.py','JOBS')==
            {'delay_test_2tab':2,'delay_test_3tab':3,'delay_test_4tab':4},'Delay jobs changed')
    # Source-derived length arithmetic: Conv1d(k=8,s=1,p=4) twice,
    # then MaxPool1d(k=8,s=4,p=4), repeated four times (dilation=1).
    source=(ROOT/'code/models.py').read_text()
    for fragment in ('kernel_size = [0, 8, 8, 8, 8]', 'pool_stride_size = [0, 4, 4, 4, 4]',
                     'filter_num = [0, 32, 64, 128, 256]', 'tgt = torch.zeros_like(query_embed)',
                     'return x.transpose(1, 2)'):
        require(fragment in source,f'Model source changed: {fragment}')
    length=30720; lengths=[]
    for _ in range(4):
        for _ in range(2): length=(length+2*4-(8-1)-1)//1+1
        length=(length+2*4-(8-1)-1)//4+1
        lengths.append(length)
    require(lengths==[7681,1921,481,121],'Unexpected convolution output lengths')
    paths=list((ROOT/'code').glob('*.py'))+[ROOT/'docs/TMWF_REPRODUCTION_REPORT.md',
        ROOT/'complete_paper_version_of_TMWF.pdf']+list((ROOT/'docs/figures').glob('tmwf_table*_comparison.png'))
    print('PASS: source configuration/job checks; analytical DFNet lengths',lengths)
    return {p:sha(p) for p in paths}


class Canvas:
    def __init__(self): self.c=[]
    def text(self,t,x,y,size=32,color=INK,bold=False,align='left'):
        self.c.append(dict(kind='text',text=t,x=x,y=y,size=size,color=color,bold=bold,align=align))
    def rect(self,x,y,w,h,color):self.c.append(dict(kind='rect',x=x,y=y,w=w,h=h,color=color))
    def line(self,x,y,x2,y2,color=MUTED,width=3):
        self.c.append(dict(kind='line',x=x,y=y,x2=x2,y2=y2,color=color,width=width))
    def arrow(self,x,y,x2,y2,color=MUTED):
        self.line(x,y,x2,y2,color,4)
        angle=math.atan2(y2-y,x2-x)
        for d in (-0.48,0.48):
            self.line(x2,y2,x2-21*math.cos(angle+d),y2-21*math.sin(angle+d),color,4)
    def box(self,x,y,w,h,title,lines,color=BLUE,size=31):
        self.rect(x,y,w,h,color)
        self.rect(x+3,y+3,w-6,h-6,'#F5F8FA' if color==BLUE else '#FFF8F1' if color==ORANGE else '#F5F6F7')
        self.rect(x,y,8,h,color)
        self.text(title,x+25,y+21,35,color,bold=True)
        for i,line in enumerate(lines):self.text(line,x+25,y+78+i*44,size)
    def title(self,t,sub):
        self.text(t,100,65,54,bold=True);self.text(sub,100,140,34,MUTED)


def architecture():
    c=Canvas();c.title('TMWF · Direction-only architecture','Original TMWF_DFNet implementation · B = batch size · tensor axes shown explicitly')
    c.box(100,240,1150,205,'Tor-cell direction sequence',
          ['+1 outbound · −1 inbound · 0 padding','DataGenerator: truncate / zero-pad → (B, 30,720)'],size=32)
    c.box(1400,240,550,205,'Learned positions',
          ['121 × 256 embeddings','Encoder Q/K; decoder memory K'],size=28)
    c.box(2050,240,550,205,'Six learned tab queries',
          ['6 × 256; repeated per batch','Initial decoder target = zeros'],size=29)
    c.arrow(375,445,375,570,BLUE)
    c.arrow(1675,445,1675,570,BLUE)
    c.arrow(2325,445,2325,570,BLUE)
    # Positions also enter decoder cross-attention keys through the pos argument.
    c.text('Shared positional encoding also conditions decoder cross-attention keys.',100,1810,29,MUTED)
    c.box(100,570,550,320,'DFNet backbone',
          ['Input unsqueeze: (B, 1, 30,720)','4 convolution / pooling blocks','Channels: 32 → 64 → 128 → 256','Output: (B, 121, 256)'],size=29)
    c.box(750,570,550,320,'Feature projection',
          ['Linear: 256 → 256','Layer normalization','Embedding dimension: 256','Output: (B, 121, 256)'])
    c.box(1400,570,550,320,'Transformer encoder',
          ['2 layers · 8 attention heads','Self-attention + feedforward','Feedforward dimension: 1,024','Memory: (B, 121, 256)'],size=29)
    c.box(2050,570,550,320,'Transformer decoder',
          ['2 layers · 8 attention heads','Self- and cross-attention','Feedforward dimension: 1,024','Output: (B, 6, 256)'],size=29)
    for x in (650,1300,1950):c.arrow(x,730,x+100,730,BLUE)
    c.arrow(2325,890,2325,1070,BLUE)
    c.box(2050,1070,550,250,'Shared classification head',
          ['Linear: 256 → 51 per slot','No extra softmax in forward()','Output logits: (B, 6, 51)'],size=29)
    c.box(1400,1070,550,250,'Six ordered logit slots',
          ['51 class scores in each slot','Logits: (B, 6, 51)','Argmax only at evaluation'],size=29)
    c.arrow(2050,1195,1950,1195,BLUE)
    c.box(750,1070,550,250,'Training loss',
          ['Sum of 6 cross-entropy terms','Logit slot i ↔ target slot i','No Hungarian matching'],color=ORANGE,size=29)
    c.arrow(1400,1195,1300,1195,ORANGE)
    c.box(100,1070,550,250,'Ordered target labels',
          ['Shape: (B, 6)','Raw tab labels in original order','Pad missing slots with class 50'],color=ORANGE,size=29)
    c.arrow(650,1195,750,1195,ORANGE)
    c.text('Loss consumes the logits, not the argmax labels.',750,1365,30,MUTED)
    c.box(100,1510,1200,250,'Label semantics',
          ['Classes 0–49: monitored websites','Class 50: unmonitored website OR unused tab slot','Correct no-tab predictions are excluded by Overall metrics.'],color=MUTED,size=31)
    c.box(1400,1510,1200,250,'Representation and scope',
          ['Direction-only model: timestamps are not input features.','Dropout argument: 0.1; no architecture changes.','Blue: model computation · Orange: training supervision'],color=MUTED,size=31)
    c.text('Source: models.py (DFNet, Transformer, TMWF_DFNet); simulation_openworld.DataGenerator;',100,1880,29,MUTED)
    c.text('train_appendix_f.py / train_appendix_h.py. Paper: architecture §4 and Table 1.',100,1925,29,MUTED)
    c.text('Shapes derived from layer definitions; no model forward pass was run to create this diagram.',100,1990,29,MUTED)
    return c.c


def pipeline():
    c=Canvas();c.title('TMWF · Experimental pipeline','Separate Appendix F and H models · preserved source partitions · final epoch-50 evaluation')
    c.rect(100,210,36,27,BLUE);c.text('Training / weight updates',155,204,30)
    c.rect(620,210,36,27,ORANGE);c.text('Validation or evaluation / no weight updates',675,204,30)
    c.text('Checkpoint tags F50 / H50 identify frozen model inputs to evaluation; arrows carry traces.',100,258,27,MUTED)
    c.box(100,330,580,470,'Original single-tab TRAIN',
          ['TBB train partition','8,000 source traces','4,000 monitored','4,000 unmonitored','Used only for synthesis','of training mixtures'],size=31)
    c.box(880,330,810,200,'Appendix F · duration synthesis',
          ['20,000 six-tab training traces','Original merger · expand_rate = 5'],size=33)
    c.box(880,600,810,200,'Appendix H · duration synthesis',
          ['40,000 six-tab training traces','Original merger · expand_rate = 10'],size=33)
    c.arrow(680,430,880,430,BLUE);c.arrow(680,700,880,700,BLUE)
    c.box(1890,330,710,200,'F50 · separate model from scratch',
          ['50 epochs → epoch_050.pth','Appendix F final checkpoint'],size=31)
    c.box(1890,600,710,200,'H50 · separate model from scratch',
          ['50 epochs → epoch_050.pth','No Appendix F initialization'],size=31)
    c.arrow(1690,430,1890,430,BLUE);c.arrow(1690,700,1890,700,BLUE)
    c.line(100,850,2600,850,'#DDE2E6',3)
    c.box(100,900,580,700,'Original single-tab TEST',
          ['Held-out TBB partition','2,000 source traces','1,000 monitored','1,000 unmonitored','','No optimizer updates','from these mixtures'],color=ORANGE,size=31)
    entries=[(900,'F · duration-synthetic tests',
              ['5,000 each: 2 / 4 / 6 tabs','6-tab set also monitors training'],
              'F50 → Table 7 evaluation',['Basic + Advanced metrics','Each tab count scored separately']),
             (1150,'H · ratio-synthetic tests',
              ['1,000 each: 2 / 3 / 4 tabs','Mixed set also monitors training'],
              'H50 → mixed ratio results',['3,000 samples scored jointly','Original Overall Basic metrics']),
             (1400,'H · copy-isolated delay synthesis',
              ['1,000 each: 2 / 3 / 4 tabs','Documented deviation; evaluation only'],
              'H50 → mixed delay results',['3,000 samples scored jointly','Not literal released-code execution'])]
    for y,title,lines,target,notes in entries:
        c.box(880,y,810,200,title,lines,color=ORANGE,size=31)
        c.arrow(680,y+100,880,y+100,ORANGE)
        c.box(1890,y,710,200,target,notes,color=ORANGE,size=30)
        c.arrow(1690,y+100,1890,y+100,ORANGE)
    c.box(100,1650,580,200,'Original TBB real files',
          ['2tabs / 3tabs / 4tabs','Separate real capture source'],color=ORANGE,size=30)
    c.box(880,1650,810,200,'Real-world evaluation subsets',
          ['First 1,000 each: 2 / 3 / 4 tabs','Evaluation only; excludes 2tabs-extra'],color=ORANGE,size=31)
    c.box(1890,1650,710,200,'H50 → real-world results',
          ['3,000 jointly + 1,000 per tab count','Reuse the same predictions'],color=ORANGE,size=29)
    c.arrow(680,1750,880,1750,ORANGE);c.arrow(1690,1750,1890,1750,ORANGE)
    c.text('Monitoring does not update weights: no early stopping or best-model selection; epoch 50 is fixed.',100,1905,29,MUTED)
    c.text('F50: outputs/appendix_f/full/run_97g8i7wd/epoch_050.pth',100,1950,28,MUTED)
    c.text('H50: outputs/appendix_h/full/run_8mrp3k_2/epoch_050.pth',100,1993,28,MUTED)
    return c.c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();before=validate()
    if args.check:return
    paths=[ROOT/'docs/figures'/name for name in ('tmwf_architecture.png','tmwf_experimental_pipeline.png')]
    for p in paths:
        require(not p.exists() and not p.is_symlink(),f'Refusing overwrite: {p}')
        require(not p.parent.is_symlink(),'Symlink output directory refused')
    images=[render(architecture()),render(pipeline())]
    require(all(sha(p)==h for p,h in before.items()),'Existing source/report/figures changed')
    for p,b in zip(paths,images):
        with p.open('xb') as f:f.write(b)
        print(f'CREATED {p.relative_to(ROOT)}: 2700 × 2100 px, 300 dpi, {len(b):,} bytes')
    print('PASS: checked text extents; source and pre-existing figure hashes unchanged. Inspect PNG layout visually.')


if __name__=='__main__':main()
