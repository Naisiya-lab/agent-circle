"""Exercise indexing, extraction and coordinate mapping without any model/API."""
from html import escape
import json
from pathlib import Path

from table_tools import TableTools

BASE = Path(__file__).resolve().parent

EXAMPLE_HTML = '''<table><caption>Tool example</caption>
<thead><tr><th>Item</th><th colspan="2">Values</th></tr></thead>
<tbody><tr><td rowspan="2">A</td><td>4005000000</td><td></td></tr>
<tr><td style="background-color:yellow" data-highlight="true">95</td><td>22.1%</td></tr>
<tr><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td></tr>
<tr><td>B</td><td>4005000000</td><td></td></tr></tbody></table>'''


def run_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    understanding = {"tables": [{"id": "t1", "html": EXAMPLE_HTML}]}
    context = {"sample_id": "external_tools_example", "understanding": understanding}
    service = TableTools(context["sample_id"], understanding, state_dir=output/"state")
    version = service.table_version
    calls = []
    def call(role, tool, args, filename):
        request = {"tool": tool, "arguments": args}
        (output/filename).write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        response = service.call(role, tool, args)
        assert response["ok"], response
        calls.append({"role": role, "request": request, "response": response})
        return response["data"]
    catalog = call("select", "list_tables", {"table_version": version}, "index_request.json")
    extracted = call("select", "extract_subtables", {"table_version": version, "regions": [
        {"id": "data", "source_table_id": "t1", "rows": [3, 3], "columns": [1, 2]},
        {"id": "header", "source_table_id": "t1", "rows": [1, 1], "columns": [2, 2]}]}, "extract_request.json")
    mapped = call("solve", "map_coordinates", {"table_version": version, "extraction_id": extracted["extraction_id"],
                 "subtable_id": "data", "direction": "to_source", "positions": [[2, 2], [2, 1]]}, "map_request.json")
    assert mapped["positions"][0]["mapped"]["position"] == [3, 2]
    assert mapped["positions"][1]["mapped"]["position"] == [3, 1]
    assert mapped["positions"][1]["mapped"]["cell_anchor"] == [2, 1]
    assert extracted["subtables"][0]["source_rows"] == [2, 3]
    assert extracted["subtables"][1]["source_columns"] == [2, 3]
    (output/"context.json").write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    (output/"results.json").write_text(json.dumps(calls, ensure_ascii=False, indent=2), encoding="utf-8")
    page = ['<!doctype html><meta charset="utf-8"><title>外部表格工具离线示例</title>',
            '<style>body{font-family:system-ui;margin:24px}iframe{width:100%;height:260px;border:1px solid #ddd}pre{white-space:pre-wrap}</style>',
            '<h1>表格索引、子表提取、坐标换算</h1><p>不调用模型；原文和位置由程序操作，未判断图像或答案正确性。</p>']
    for title, html in [("索引全表", catalog["tables"][0]["html"])] + [("程序子表 "+s["id"], s["html"]) for s in extracted["subtables"]]:
        markup = '<style>table{border-collapse:collapse}td,th{border:1px solid #888;padding:5px;min-width:50px}</style>'+html
        page.extend(["<h2>"+escape(title)+"</h2>", '<iframe sandbox="" srcdoc="'+escape(markup, quote=True)+'"></iframe>'])
    page.extend(["<h2>坐标查询</h2><pre>", escape(json.dumps(mapped, ensure_ascii=False, indent=2)), "</pre>"])
    (output/"preview.html").write_text("".join(page), encoding="utf-8")
    return {"output": str(output), "logical_size": [catalog["tables"][0]["row_count"], catalog["tables"][0]["column_count"]],
            "grid_position": mapped["positions"][1]["mapped"]["position"],
            "owning_cell_anchor": mapped["positions"][1]["mapped"]["cell_anchor"], "api_calls": 0}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(run_demo(BASE/"analysis/table_tools_demo_20261021"), ensure_ascii=False, indent=2))
