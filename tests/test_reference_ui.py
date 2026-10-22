"""Reference-mode rendering must never instantiate legacy expert images in text mode."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_text_reference_rendering_has_no_images_and_legacy_switch_still_works():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the reference DOM check")
    app = Path(__file__).resolve().parents[1] / "src" / "split_cloth" / "static" / "app.js"
    script = r'''
const fs = require("fs"), vm = require("vm"), assert = require("assert");
class Element {
  constructor(tag="div") { this.tagName=tag.toUpperCase(); this.children=[]; this._text=""; this.style={}; this.classList={toggle(){},add(){}}; }
  set textContent(value) { this._text=String(value); this.children=[]; }
  get textContent() { return this._text+this.children.map(child=>child.textContent||"").join(""); }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this._text=""; this.children=items; }
  get childNodes() { return this.children; }
}
const elements=new Map(), createdImages=[];
const document={
  getElementById(id) { if (!elements.has(id)) elements.set(id,new Element()); return elements.get(id); },
  createElement(tag) { const node=new Element(tag); if(tag==="img")createdImages.push(node); return node; },
  createTextNode(text) { const node=new Element(); node.textContent=text; return node; }
};
const context=vm.createContext({document, location:{origin:"http://localhost"}, URL, URLSearchParams});
const source=fs.readFileSync(process.argv[2],"utf8").split('$("run-select").addEventListener')[0];
vm.runInContext(source,context);
const api=vm.runInContext("({S,renderExperts})",context);
const criteria={schema_version:"text_stage_criteria_v1", stages:[{stage_id:1,name:"第一侧纵折",action:"第一侧实际折入",completion_criteria:["折边形成并完全松爪"],exclude:["抖开和铺平不是纵折"]}],temporal_rules:["准备先于折叠"],boundary_rule:"422结束，423进入下一阶段",provenance:{model:"GPT-test",source_document:"规范5.2"},limitations:["不能视作人工金标"]};
api.renderExperts({name:"旧示范", pages:[{url:"/legacy.png"}]});
assert.equal(createdImages.length,1);
assert.equal(document.getElementById("reference-title").textContent,"Expert 示范");
assert.equal(document.getElementById("expert-select").hidden,false);
// Even an accidental legacy pages field in this response cannot load images.
api.renderExperts({reference_mode:"text_stage_criteria",stage_criteria:criteria,pages:[{url:"/must-not-load.png"}]});
assert.equal(createdImages.length,1);
assert.equal(document.getElementById("reference-title").textContent,"文字阶段判据");
assert.equal(document.getElementById("expert-select").hidden,true);
const target=document.getElementById("expert-content");
for(const text of ["GPT 对参考图像的文字归纳","非人工金标准","第一侧实际折入","折边形成并完全松爪","抖开和铺平不是纵折","准备先于折叠","422结束，423进入下一阶段","GPT-test","规范5.2","不能视作人工金标"])assert(target.textContent.includes(text));
const hasImage=node=>node.tagName==="IMG"||node.children.some(hasImage);
assert.equal(hasImage(target),false);
api.renderExperts({pages:[{url:"/legacy-again.png"}]});
assert.equal(createdImages.length,2);
assert.equal(document.getElementById("reference-title").textContent,"Expert 示范");
assert.equal(document.getElementById("expert-select").hidden,false);
console.log("Reference modes, text criteria, provenance, and no legacy images passed.");
'''
    result = subprocess.run([node, "-", str(app)], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr or result.stdout
