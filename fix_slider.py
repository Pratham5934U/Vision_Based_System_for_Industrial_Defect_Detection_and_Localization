import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = '        pc_thresh = st.slider(\n            "Decision Threshold (Distance Cutoff)",\n            min_value=12.0, max_value=18.0, value=14.50, step=0.1,\n            help="Learned baseline: Good parts average ~13.3, Defects average ~16.8. Default 14.50 cleanly separates good from defects.",\n            key="slider_pc_thresh",\n        )'

new = '        pc_thresh = st.slider(\n            "Decision Threshold (Raw Distance Cutoff)",\n            min_value=11.0, max_value=19.0, value=14.8, step=0.1,\n            help=(\n                "Measured distributions: Good parts mean=13.3 max=15.5 | "\n                "Scratch mean=15.5 | Bent mean=17.0 | Color mean=16.1. "\n                "Default 14.8 gives best separation."\n            ),\n            key="slider_pc_thresh",\n        )'

if old in content:
    content = content.replace(old, new)
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Slider updated successfully')
else:
    print('ERROR: old text not found')
    idx = content.find('slider_pc_thresh')
    print(repr(content[max(0,idx-300):idx+100]))
