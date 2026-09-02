import json, os
base = "/Users/brianlo/ulam-research/paper"
data = json.load(open("data/paper_data.json"))
parts = ["head.part","body1.part","body2.part","body3.part","charts.part","charts2.part","charts3.part"]
html = []
html.append(open(f"{base}/head.part").read())
html.append(f'<script>window.PAPER_DATA = {json.dumps(data, separators=(",",":"))};</script>\n')
for p in parts[1:]:
    html.append(open(f"{base}/{p}").read())
out = "".join(html)
open(f"{base}/index.html","w").write(out)
print("built", len(out), "bytes ->", f"{base}/index.html")
