"""Build a dependency-free, file:// compatible tabbed trial viewer."""
import json
from pathlib import Path


def build_ui(directory,summary,entries,outcomes,intro,phases,limitations,skills,screenshots):
    def read(path):return json.loads(path.read_text())
    steps=[]
    for i,(entry,outcome) in enumerate(zip(entries,outcomes)):
        steps.append(dict(number=i+1,request=read(directory/'requests'/f'{i:03}.json'),
            observation=read(directory/'observations'/f'{i:03}.json'),commitment=entry,outcome=outcome,
            before_image=screenshots['screenshots'][i],after_image=screenshots['screenshots'][i+1]))
    data=dict(summary=summary,steps=steps,intro=intro,phases=phases,limitations=limitations,skills=skills)
    if (directory/'planner_context.txt').exists():
        data['planner_context']=(directory/'planner_context.txt').read_text()
    payload=json.dumps(data,ensure_ascii=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    template=(Path(__file__).parent/'report_templates/llm_viewer.html').read_text()
    (directory/'report.html').write_text(template.replace('__REPORT_DATA__',payload))
