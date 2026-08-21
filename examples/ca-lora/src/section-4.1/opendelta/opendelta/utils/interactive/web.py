from bigmodelvis import Visualization
import web
import re, os

space = "&nbsp;"
prefix0 = space * 9
prefix1 = f"│"+space*5
prefix2 = f"├─{space}"
prefix3 = f"└─{space}"

def colorfy(label):
    i = 0
    res = ""
    while i < len(label):
        if label[i] == '[':
            color = ""
            i += 1
            while label[i] != ']':
                color += label[i]
                i += 1
            i += 1
            if color[0].isdigit(): # dims but not color
                res += f'[{color}]'
            else:
                if res != "": res += '</span>'
                res += f'<span style="color: {color}">'
        else:
            res += label[i]
            i += 1
    res += '</span>'
    return res

compressed_pattern_1 = re.compile("[0-9]+-[0-9]+")
compressed_pattern_2 = re.compile(".+(,.+)+")

def expand_part(part):
    res = []
    if compressed_pattern_1.fullmatch(part):
        st, ed = map(int, part.split('-'))
        for i in range(st, ed+1):
            res.append( str(i) )
    elif compressed_pattern_2.fullmatch(part):
        for c in part.split(','):
            res.append( c )
    else:
        res.append( part )
    return res

def dfs(o, depth, last, old_name):
    html = ""
    module_names = expand_part(o.module_name)
    if depth > 0:
        old_last_1 = last[-1]
    if len(module_names) > 1:
        module_names = [o.module_name] + module_names
    for ith, module_name in enumerate(module_names):
        if ith == 0:
            html += f'<div>'
        elif ith == 1:
            html += f'<div class="expandable-sibling">'

        for i in range(depth-1):
            html += prefix0 if last[i] else prefix1
        if depth > 0:
            last[-1] = old_last_1 & (ith == 0 or ith == len(module_names)-1)
            html += prefix3 if last[-1] else prefix2
        length = len(o.children)
        if length > 0:
            html += f'<button class="collapsible button_inline">[+]</button>'
        name = old_name + module_name
        if ith > 0:
            label = f'[red]{module_name}{o.label[o.label.index("[",1):]}'
        else:
            label = o.label
        html += f'<button class="selectable button_inline" id="{name}">{colorfy(label)}</button>'
        if len(module_names) > 1 and ith == 0:
            html += '<button class="expandable button_inline">[*]</button>'
        html += '<br/>'
        html += f'<div class="content">'
        for i, child in enumerate(o.children):
            last = last + [i == length-1]
            html += dfs(child, depth+1, last, name + ".")
            last.pop()

        html += "</div>"
        if ith == 0 or (ith > 1 and ith == len(module_names)-1):
            html += "</div>"
    return html

urls = (
    '/submit/(.*)', 'submit',
    '/(.*)', 'hello',
)

class PortApplication(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ('0.0.0.0', port))

app = PortApplication(urls, globals())
render = web.template.render(os.path.join(os.path.dirname(__file__), 'templates/'))
names = []

class hello:
    def GET(self, name):
        return render.index(content=html)
class submit:
    def GET(self, _):
        global names
        names = [name[5:] for name in web.input(name=[]).name]
        app.stop()

def interactive(model, port=8888):
    tree = Visualization(model).structure_graph(printTree=False)

    global html
    html = dfs(tree, 0, [], "")

    print()
    print("If on your machine, open the link below for interactive modification.\n "
    "If on remote host, you could use port mapping, "
    "or run in vscode terminal, which automatically do port mapping for you.")
    app.run(port)
    global names
    print("modified_modules:")
    print(names)
    return names
