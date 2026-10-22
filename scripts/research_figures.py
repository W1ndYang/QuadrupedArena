"""Publication-style descriptive figures; one episode per condition, no inference."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from research_data import OUT

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
    'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
    'lines.linewidth':1.3,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
    'savefig.facecolor':'white','figure.facecolor':'white','axes.labelcolor':'#24343d',
    'text.color':'#24343d','xtick.color':'#24343d','ytick.color':'#24343d'})
INK='#24343d'; GREY='#8b9298'; RED='#b84e5d'
SKILLS={'CAPTURE_POINT':'#007F73','MOVE_TO':'#3979B5','TAKE_COVER':'#A55E18','AIM_AT_TRACK':'#B85D86','OBSERVE':'#79824A'}

def panel(ax,letter,title):
    ax.set_title(title,loc='left',pad=10)
    ax.text(-.13,1.07,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)
def tidy(ax):ax.grid(axis='y',color='#e6e9e9',lw=.5);ax.set_axisbelow(True)
def save(fig,name,rows):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    for ext in ['svg','pdf','png']:fig.savefig(folder/f'{name}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.08)
    (folder/f'{name}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    plt.close(fig)
def series(ax,r,label=None):
    s=r['timeline'];ax.plot([x['t'] for x in s],[x['score'] for x in s],color=r['color'],label=label or r['short'])
    m=r['metric'];ax.scatter(m['end_seconds'],m['score_observed'],s=25,zorder=5,
        facecolor=r['color'] if m['completed'] else 'white',edgecolor=r['color'],linewidth=1)
def collapse(ax):ax.axvline(20,color=GREY,lw=.7,ls='--',zorder=0)
def intervals(ax,r,field,y,color,height=.65):
    states=r['timeline'];start=None
    for prev,s in zip(states[:-1],states[1:]):
        active=field(s)
        if active and start is None:start=prev['t']
        if not active and start is not None:ax.broken_barh([(start,prev['t']-start)],(y-height/2,height),color=color);start=None
    if start is not None:ax.broken_barh([(start,states[-1]['t']-start)],(y-height/2,height),color=color)

def build():
    d=json.loads((OUT/'data/analysis.json').read_text());allr=d['runs'];vision=allr[2:]
    bt=d['baseline'];last=bt['timeline'][-1]
    baseline={**bt,'short':'Rule / A*','metric':dict(end_seconds=last['t'],score_observed=last['score'],completed=True)}
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.4),layout='constrained')
    for ax,runs,letter,title in [(axs[0,0],allr[:2]+[baseline],'a','Oracle cohort + contextual baseline'),(axs[0,1],vision,'b','Vision + semantic skills cohort')]:
        for i,r in enumerate(runs):
            m=r['metric'];ax.scatter(m['score_observed'],i,s=45,facecolor=r['color'] if m['completed'] else 'white',edgecolor=r['color'],zorder=3)
            ax.text(m['score_observed']+.35,i,f"{m['score_observed']:g}"+(' *' if not m['completed'] else ''),va='center',fontsize=8)
        ax.set_yticks(range(len(runs)),[r['short'] for r in runs]);ax.invert_yaxis();ax.set_ylim(len(runs)-.5,-.5);ax.set_xlim(0,14.5);ax.set_xlabel('Observed score (points)');panel(ax,letter,title);ax.grid(axis='x',color='#e6e9e9',lw=.5)
    for r in allr[:2]+[baseline]:series(axs[1,0],r)
    for r in vision:series(axs[1,1],r)
    for ax,letter,title in [(axs[1,0],'c','Oracle score trajectories'),(axs[1,1],'d','Vision score trajectories')]:
        collapse(ax);ax.set(xlim=(0,40),ylim=(-.3,14.5),xlabel='Simulation time (s)',ylabel='Cumulative score (points)');tidy(ax);panel(ax,letter,title);ax.legend(loc='upper left',frameon=False,ncol=1)
    save(fig,'fig01_outcomes',{'metrics':d['metrics'],'baseline_end':last})

    fig,axs=plt.subplots(2,2,figsize=(7.2,6.2),layout='constrained')
    ax=axs[0,0]
    for i,r in enumerate(allr):
        vals=[s['wait_seconds'] for s in r['steps']];jitter=np.linspace(-.16,.16,len(vals))
        ax.scatter(vals,np.full(len(vals),i)+jitter,c=r['color'],s=10,alpha=.7,linewidths=0)
        ax.plot([np.median(vals)]*2,[i-.25,i+.25],c=INK,lw=1.5)
    ax.axvline(2,c=RED,ls=':',lw=1);ax.set_yticks(range(7),[r['id'] for r in allr]);ax.invert_yaxis();ax.set(xlabel='External decision wait (wall-clock s)',xlim=(-3,220));panel(ax,'a','Accepted decisions: dots; median: bar')
    ax=axs[0,1]
    for i,r in enumerate(vision[:4]):
        for s in r['steps']:
            ax.plot([s['first_output_seconds'],s['api_seconds']],[i,i],color=r['color'],alpha=.22,lw=1)
            ax.scatter(s['first_output_seconds'],i-.10,s=10,c=r['color'],marker='|');ax.scatter(s['api_seconds'],i+.10,s=10,c=r['color'])
    ax.set_yticks(range(4),[r['id'] for r in vision[:4]]);ax.invert_yaxis();ax.set(xlabel='API elapsed (s)',xlim=(0,160));panel(ax,'b','First output | to full response ●')
    ax=axs[1,0];rows=[]
    for i,r in enumerate([allr[1]]+vision[:4]):
        u=r['usage'];prompt=u['prompt_tokens']/1000;completion=u['completion_tokens']/1000;reason=(u.get('reasoning_tokens') or 0)/1000
        ax.barh(i,prompt,color='#bfd1de',height=.6);ax.barh(i,reason,left=prompt,color='#6b7395',height=.6);ax.barh(i,completion-reason,left=prompt+reason,color='#007F73',height=.6)
        ax.text((u['total_tokens']/1000)+5,i,f"{u['total_tokens']/1000:.1f}",va='center',fontsize=7);rows.append({'id':r['id'],**u})
    ax.set_yticks(range(5),[f'{r["id"]} ({r["metric"]["calls"]} calls)' for r in [allr[1]]+vision[:4]]);ax.invert_yaxis();ax.set(xlabel='Provider-reported tokens (thousands)',xlim=(0,650));panel(ax,'c','Primary runs only; historical costs separate')
    ax.legend(handles=[Rectangle((0,0),1,1,color=c,label=l) for c,l in [('#bfd1de','Prompt'),('#6b7395','Reasoning subset'),('#007F73','Other completion')]],frameon=False,loc='lower right',fontsize=6.5)
    ax=axs[1,1]
    for i,r in enumerate(vision):
        vals=[s['compile_seconds'] for s in r['steps']];ax.scatter(vals,np.full(len(vals),i)+np.linspace(-.16,.16,len(vals)),s=12,c=r['color'],alpha=.7,linewidths=0)
    ax.axvline(2,c=RED,ls=':',lw=1);ax.text(1.94,4.5,'2 s control period',ha='right',va='center',fontsize=7,color=RED)
    ax.set_yticks(range(5),[r['id'] for r in vision]);ax.set(xlim=(0,2.12),ylim=(4.9,-.5),xlabel='Skill compilation (s)');panel(ax,'d','Planning backend is not the main delay')
    save(fig,'fig02_latency_tokens',{'steps':[{k:s[k] for k in ['run','decision','wait_seconds','api_seconds','first_output_seconds','compile_seconds']} for r in allr for s in r['steps']],'usage':rows,'historical_costs_excluded':True})

    fig,axs=plt.subplots(3,2,figsize=(7.2,8.1),layout='constrained')
    cfg=vision[0]['config']
    for ax,r,letter in zip(axs.flat,vision,'abcde'):
        for w in cfg['arena']['covers']:
            x,y=w['center'];sx,sy=w['size'][:2];ax.add_patch(Rectangle((x-sx/2,y-sy/2),sx,sy,fc='#d1d6d8',ec='#7f888d',lw=.6))
        w=cfg['collapse']['debris'];x,y=w['center'];sx,sy=w['size'][:2];ax.add_patch(Rectangle((x-sx/2,y-sy/2),sx,sy,fc='none',ec='#a59a7d',lw=.7,hatch='///'))
        for p in cfg['arena']['task_points']:
            ax.add_patch(Circle(p['center'],p['radius'],fill=False,ec='#73854b',lw=1));ax.text(p['center'][0]+.67,p['center'][1]-.15,p['name'],fontsize=7,ha='left',va='center')
        xy=np.array([s['xy'] for s in r['timeline']]);ax.plot(xy[:,1,0],xy[:,1,1],c=GREY,ls='--',lw=.9);ax.plot(xy[:,0,0],xy[:,0,1],c=r['color'],lw=1.4)
        ax.scatter(*xy[0,0],marker='s',s=15,c=r['color']);ax.scatter(*xy[-1,0],s=30,facecolor=r['color'] if r['metric']['completed'] else 'white',edgecolor=r['color'],zorder=4)
        for ev in r['events']:
            if ev['type']=='hit' and ev['target']==0:
                st=min(r['timeline'],key=lambda s:abs(s['t']-ev['time']));ax.scatter(*st['xy'][0],marker='x',s=24,c=RED,zorder=5)
        for t in [8,16,24,32]:
            if t<r['metric']['end_seconds']:
                st=min(r['timeline'],key=lambda s:abs(s['t']-t));ax.scatter(*st['xy'][0],s=5,c=INK);ax.annotate(str(t),st['xy'][0],xytext=(3,3),textcoords='offset points',fontsize=6)
        ax.set(xlim=(-6,6),ylim=(-4,4),xlabel='x (m)',ylabel='y (m)',aspect='equal');panel(ax,letter,r['short']+(' — interrupted' if not r['metric']['completed'] else ''))
    ax=axs[2,1];ax.axis('off');panel(ax,'f','Posthoc ground truth — not planner input')
    lines=[Line2D([0],[0],color='#007F73',label='Quadruped A trajectory'),Line2D([0],[0],color=GREY,ls='--',label='Humanoid B trajectory'),Line2D([0],[0],marker='s',color='none',markerfacecolor=INK,label='Start'),Line2D([0],[0],marker='x',color=RED,ls='none',label='A receives a hit'),Line2D([0],[0],marker='o',color=INK,ls='none',label='Completed endpoint')]
    ax.legend(handles=lines,loc='upper left',frameon=False,fontsize=7.5,labelspacing=.55)
    ax.text(0,.06,'Hatched footprint: collapsed High3 at 20 s.\nLabels on path: simulation seconds.\nSame geometry and scale in all panels.\nPaths stop at the observed terminal event.',transform=ax.transAxes,fontsize=7.3,linespacing=1.6)
    save(fig,'fig03_trajectories',{'timeline_source':'data/analysis.json','config':cfg,'runs':[r['id'] for r in vision]})

    fig,axs=plt.subplots(3,1,figsize=(7.2,7.2),layout='constrained')
    for i,r in enumerate(vision):
        for s in r['steps']:
            axs[0].broken_barh([(s['time'],s['end']-s['time'])],(i-.32,.64),facecolor=SKILLS[s['skill']],edgecolor='white',lw=.4)
            if s['compiled_state']=='UNSAFE_ROUTE':axs[0].scatter(s['time']+min(1,(s['end']-s['time'])/2),i,s=25,marker='x',c='#efcf73',zorder=4)
        for ep in d['episodes']:
            if ep['run']==r['id']:
                axs[1].broken_barh([(ep['start'],ep['end']-ep['start'])],(i-.32,.64),facecolor=r['color'],alpha=.85)
                axs[1].text((ep['start']+ep['end'])/2,i,f"P{ep['point']}\n{ep['points']:g}",color='white',ha='center',va='center',fontsize=6.5)
                if not ep['closed']:axs[1].scatter(ep['end'],i,s=25,facecolor='white',edgecolor=r['color'],zorder=5)
        intervals(axs[2],r,lambda s:s['human_visible'],i,'#d9dee1')
        intervals(axs[2],r,lambda s:s['human_visible'] and s['speed'][0]<.05,i,r['color'],height=.40)
        for e in r['events']:
            if e['type']=='hit' and e['target']==0:
                for ax in axs[1:]:ax.scatter(e['time'],i,s=28,c=RED,marker='x',zorder=5)
        for ax in axs:ax.plot([r['metric']['end_seconds']]*2,[i-.4,i+.4],c=INK,lw=.7,ls='-' if r['metric']['completed'] else ':')
    for ax,letter,title in zip(axs,'abc',['Requested skills (× = rejected route → HOLD)','Triggered target windows; labels = awarded points','B sees A (grey); A also stationary (colour); × = hit']):
        ax.set_yticks(range(5),[r['short'] for r in vision]);ax.set(xlim=(0,40),ylim=(4.6,-.6),xlabel='Simulation time (s)');collapse(ax);panel(ax,letter,title)
    axs[0].legend(handles=[Rectangle((0,0),1,1,color=v,label=k.replace('_',' ')) for k,v in SKILLS.items()],ncol=3,frameon=False,loc='upper left',bbox_to_anchor=(0,-.28),fontsize=6.2)
    save(fig,'fig04_skill_execution',{'episodes':d['episodes'],'skills':{r['id']:[{'time':s['time'],'end':s['end'],'skill':s['skill'],'compiled_state':s['compiled_state']} for s in r['steps']] for r in vision}})

    fig,axs=plt.subplots(2,2,figsize=(7.2,5.8),layout='constrained');x=np.arange(5)
    ax=axs[0,0];ax.bar(x-.17,[r['metric']['unique_positive_frame_claims'] for r in vision],width=.34,color='#bfd1de',label='Unique positive frame claims');ax.bar(x+.17,[r['metric']['annotations'] for r in vision],width=.34,color='#007F73',label='Foot annotations submitted');ax.set_xticks(x,[r['id'] for r in vision]);ax.set_ylabel('Recorded count');ax.set_ylim(0,max(r['metric']['unique_positive_frame_claims'] for r in vision)*1.4+2);ax.legend(frameon=False,fontsize=6.5);tidy(ax);panel(ax,'a','Description ≠ executable localization')
    ax=axs[0,1]
    for i,r in enumerate(vision):
        v=[a['error_m'] for a in r['annotations']]
        if v:ax.scatter(np.full(len(v),i)+np.linspace(-.12,.12,len(v)),v,s=28,c=r['color'],edgecolor='white',lw=.4)
        else:ax.text(i,.026,'No\nannotations',ha='center',va='center',fontsize=6.5)
    ax.set_yscale('log');ax.set_xticks(x,[r['id'] for r in vision]);ax.set(xlim=(-.5,4.5),ylim=(.015,6),ylabel='Ground-projection discrepancy (m)');tidy(ax);panel(ax,'b','Each annotation; not detection accuracy')
    ax=axs[1,0]
    for i,r in enumerate(vision):
        ax.barh(i,r['metric']['human_sees_a_seconds'],height=.6,color='#d9dee1');ax.barh(i,r['metric']['stationary_seen_seconds'],height=.32,color=r['color'])
    ax.set_yticks(x,[r['id'] for r in vision]);ax.invert_yaxis();ax.set(xlim=(0,max(r['metric']['human_sees_a_seconds'] for r in vision)*1.12),xlabel='Observed exposure time (s)');panel(ax,'c','B sees A: total vs stationary subset')
    ax=axs[1,1]
    for i,r in enumerate(vision):
        v=r['metric']['score_at_16_8'];ax.scatter(i,v,c=r['color'],s=40);ax.text(i,v+.3,f'{v:g}',ha='center',fontsize=8)
    ax.set_xticks(x,[r['id'] for r in vision]);ax.set(ylim=(0,9),ylabel='Score at 16.80 s (points)');tidy(ax);panel(ax,'d','Common observed horizon; n = 1 each')
    save(fig,'fig05_perception',{'annotations':d['annotations'],'metrics':[r['metric'] for r in vision]})

    fig,axs=plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained');ax=axs[0]
    t=np.linspace(0,4,300)
    for tau,color,label in [(1,'#007F73','A: τ = 1.0 s'),(.5,RED,'B: τ = 0.5 s')]:
        ax.plot(t,100*(1-np.exp(-1.5*(t-tau*(1-np.exp(-t/tau))))),c=color,label=label)
    ax.set(xlabel='Uninterrupted valid aim (s)',ylabel='Hit probability by time t (%)',xlim=(0,4),ylim=(0,100));ax.legend(frameon=False);tidy(ax);panel(ax,'a','Known rule model, not fitted data')
    ax=axs[1];r=vision[0];tokens=[s['api']['usage']['completion_tokens'] for s in r['steps']]
    ax.plot(range(1,len(tokens)+1),np.array(tokens)/1000,c=r['color'],marker='o',ms=3)
    ax.axhline(16.384,c=RED,ls='--',lw=.8);ax.text(.03,.94,'Dashed: previous cap (16,384)',transform=ax.transAxes,fontsize=6.8,color=RED)
    for i,n in enumerate(tokens):
        if n>16384:
            ax.scatter(i+1,n/1000,s=34,c=r['color']);ax.annotate(f'{n:,}\nstop',(i+1,n/1000),xytext=(4,5),textcoords='offset points',fontsize=6.5)
    ax.set(xlabel='API request index (1-based)',ylabel='Completion tokens incl. reasoning (k)',xlim=(.5,20.7),ylim=(0,33));ax.set_xticks([1,5,10,15,20]);tidy(ax);panel(ax,'b','DS max rerun: all 20 returned action JSON')
    save(fig,'fig06_mechanisms',{'rule':{'lambda_max':1.5,'tau_a':1,'tau_b':.5},'completion_tokens':tokens,'previous_cap':16384,'current_max_tokens':r['profile']['max_tokens'],'historical_failure':next(h['failed_response'] for h in d['history'] if h['id']=='DH16')})

if __name__=='__main__':build();print('Wrote six SVG/PDF/600-dpi PNG figure sets.')
