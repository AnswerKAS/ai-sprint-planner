import html as _html
from dataclasses import dataclass
from datetime import datetime

from tasks.model import SprintTask


# ── data structures ────────────────────────────────────────────────────────────

@dataclass
class IterationRecord:
    iteration: int
    plan: str
    feedback: str
    validated: bool
    total_sp: float
    consultation: str  # specialist advice collected AFTER this iteration (empty on last)


@dataclass
class TeamReportData:
    team_name: str
    session_id: str
    capacity: float
    final_task_ids: list[str]
    final_tasks: list[SprintTask]
    total_sp: float
    utilization_pct: float
    total_iterations: int
    validated: bool
    final_plan: str
    agent_results: dict[str, str]
    iterations: list[IterationRecord]


# ── colour helpers ─────────────────────────────────────────────────────────────

_PRIORITY_CSS = {
    "critical": "#dc2626",
    "high": "#ea580c",
    "medium": "#d97706",
    "low": "#16a34a",
}

_CATEGORY_CSS = {
    "incident": "#dc2626",
    "task": "#2563eb",
    "project": "#7c3aed",
    "quota": "#059669",
}

_AGENT_LABELS = {
    "inc_agent": "Инцидентные задачи (inc_agent)",
    "task_agent": "Внутренние задачи (task_agent)",
    "project_agent": "Проектные задачи (project_agent)",
    "quota_agent": "Квотные задачи (quota_agent)",
}


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:12px;font-size:0.78em;font-weight:600;white-space:nowrap">'
        f'{_html.escape(str(text))}</span>'
    )


def _priority_badge(priority: str) -> str:
    color = _PRIORITY_CSS.get(priority, "#6b7280")
    return _badge(priority, color)


def _category_badge(category: str) -> str:
    color = _CATEGORY_CSS.get(category, "#6b7280")
    return _badge(category, color)


def _utilization_color(pct: float) -> str:
    if pct > 110:
        return "#dc2626"
    if pct >= 90:
        return "#16a34a"
    if pct >= 70:
        return "#d97706"
    return "#6b7280"


# ── HTML building blocks ───────────────────────────────────────────────────────

def _css() -> str:
    return """
<style>
  :root {
    --bg: #f8fafc;
    --card-bg: #ffffff;
    --border: #e2e8f0;
    --text: #1e293b;
    --muted: #64748b;
    --accent: #3b82f6;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); padding: 24px; line-height: 1.6; }
  h1 { font-size: 1.8em; font-weight: 700; margin-bottom: 4px; }
  h2 { font-size: 1.4em; font-weight: 700; margin: 32px 0 16px; border-bottom: 2px solid var(--border); padding-bottom: 8px; }
  h3 { font-size: 1.1em; font-weight: 600; margin: 20px 0 10px; }
  h4 { font-size: 0.95em; font-weight: 600; margin: 12px 0 6px; color: var(--muted); }
  .meta { color: var(--muted); font-size: 0.85em; margin-bottom: 28px; }

  /* TOC */
  .toc { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px;
         padding: 16px 24px; margin-bottom: 32px; display: inline-block; }
  .toc a { color: var(--accent); text-decoration: none; display: block; padding: 2px 0; }
  .toc a:hover { text-decoration: underline; }

  /* Summary cards */
  .cards { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }
  .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px;
          padding: 16px 20px; min-width: 160px; flex: 1; }
  .card .label { font-size: 0.78em; color: var(--muted); font-weight: 600;
                 text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
  .card .value { font-size: 1.5em; font-weight: 700; }
  .card .sub { font-size: 0.82em; color: var(--muted); margin-top: 2px; }

  /* Progress bar */
  .progress-wrap { background: #e2e8f0; border-radius: 999px; height: 14px;
                   margin-bottom: 24px; overflow: hidden; }
  .progress-bar { height: 100%; border-radius: 999px; transition: width 0.4s; }

  /* Table */
  table { width: 100%; border-collapse: collapse; font-size: 0.88em; margin-bottom: 24px; }
  thead tr { background: #f1f5f9; }
  th { padding: 10px 12px; text-align: left; font-weight: 600; color: var(--muted);
       border-bottom: 2px solid var(--border); white-space: nowrap; }
  td { padding: 9px 12px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  tbody tr:hover { background: #f8fafc; }
  tfoot td { font-weight: 700; border-top: 2px solid var(--border); background: #f1f5f9; }
  .task-id { font-family: monospace; font-size: 0.9em; color: var(--accent); font-weight: 600; }
  .num { text-align: right; }

  /* Collapsible details */
  details { border: 1px solid var(--border); border-radius: 8px; margin-bottom: 10px;
            background: var(--card-bg); overflow: hidden; }
  details > summary {
    padding: 12px 16px; font-weight: 600; cursor: pointer; user-select: none;
    list-style: none; display: flex; align-items: center; gap: 8px;
    background: #f8fafc;
  }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::before { content: "▶"; font-size: 0.7em; color: var(--muted);
                               transition: transform 0.2s; display: inline-block; }
  details[open] > summary::before { transform: rotate(90deg); }
  details > summary:hover { background: #f1f5f9; }
  .details-body { padding: 16px; border-top: 1px solid var(--border); }

  /* Pre / agent output */
  pre { background: #0f172a; color: #e2e8f0; padding: 14px 16px; border-radius: 8px;
        font-size: 0.82em; overflow-x: auto; white-space: pre-wrap; word-break: break-word;
        line-height: 1.5; margin: 0; }

  /* Iteration timeline */
  .timeline { position: relative; padding-left: 32px; }
  .timeline::before { content: ""; position: absolute; left: 10px; top: 0; bottom: 0;
                       width: 2px; background: var(--border); }
  .tl-item { position: relative; margin-bottom: 20px; }
  .tl-dot { position: absolute; left: -26px; top: 6px; width: 14px; height: 14px;
             border-radius: 50%; background: var(--accent); border: 2px solid #fff;
             box-shadow: 0 0 0 2px var(--border); }
  .tl-dot.accepted { background: #16a34a; }
  .tl-dot.rejected { background: #dc2626; }

  /* Verdict chip */
  .verdict { display: inline-block; padding: 2px 10px; border-radius: 999px;
             font-size: 0.78em; font-weight: 700; margin-left: 8px; }
  .verdict.ok { background: #dcfce7; color: #166534; }
  .verdict.fail { background: #fee2e2; color: #991b1b; }

  /* Feedback box */
  .feedback-box { background: #fff7ed; border-left: 4px solid #f97316;
                  padding: 10px 14px; border-radius: 0 6px 6px 0; font-size: 0.88em; margin: 8px 0; }
  .consult-box { background: #f0f9ff; border-left: 4px solid #38bdf8;
                 padding: 10px 14px; border-radius: 0 6px 6px 0; font-size: 0.88em; margin: 8px 0; }

  footer { margin-top: 48px; text-align: center; color: var(--muted); font-size: 0.82em; }
  .section-anchor { scroll-margin-top: 20px; }
</style>
"""


def _header(generated_at: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sprint Planning Report — {_html.escape(generated_at)}</title>
  {_css()}
</head>
<body>
<h1>Sprint Planning Report</h1>
<p class="meta">Сформирован: {_html.escape(generated_at)}</p>
"""


def _toc(reports: list[TeamReportData]) -> str:
    links = "\n".join(
        f'  <a href="#team-{_html.escape(r.team_name)}">{_html.escape(r.team_name)}'
        f' — {r.utilization_pct:.1f}% загрузки</a>'
        for r in reports
    )
    return f'<div class="toc"><strong>Команды</strong>\n{links}\n</div>\n'


def _summary_cards(report: TeamReportData) -> str:
    util_color = _utilization_color(report.utilization_pct)
    bar_pct = min(report.utilization_pct, 110)
    bar_width = bar_pct / 110 * 100

    cards_html = f"""
<div class="cards">
  <div class="card">
    <div class="label">Ёмкость команды</div>
    <div class="value">{report.capacity:.1f} SP</div>
  </div>
  <div class="card">
    <div class="label">Взято в спринт</div>
    <div class="value">{report.total_sp:.1f} SP</div>
  </div>
  <div class="card">
    <div class="label">Загрузка</div>
    <div class="value" style="color:{util_color}">{report.utilization_pct:.1f}%</div>
    <div class="sub">цель: 90–110%</div>
  </div>
  <div class="card">
    <div class="label">Итераций</div>
    <div class="value">{report.total_iterations}</div>
    <div class="sub">{"план принят" if report.validated else "план не принят"}</div>
  </div>
  <div class="card">
    <div class="label">Задач в спринте</div>
    <div class="value">{len(report.final_tasks)}</div>
  </div>
</div>
<div class="progress-wrap">
  <div class="progress-bar" style="width:{bar_width:.1f}%;background:{util_color}"></div>
</div>
"""
    return cards_html


def _task_table(report: TeamReportData) -> str:
    task_index = {t.task_id: t for t in report.final_tasks}
    rows = []
    total_sp = 0.0

    for i, tid in enumerate(report.final_task_ids, 1):
        task = task_index.get(tid)
        if task is None:
            continue
        total_sp += task.sp

        quota_mark = (
            f' <span style="color:#059669;font-size:0.8em;font-weight:600">[Q]</span>'
            if task.quota and task.quota > 0 else ""
        )
        escalation_cell = (
            f'<span style="color:#dc2626;font-weight:600">{task.escalation_count}</span>'
            if task.escalation_count > 0 else "—"
        )
        rice_cell = f"{task.rice:.0f}" if task.rice is not None else "—"
        stage_cell = _html.escape(task.stage) if task.stage else "—"
        customer_cell = _html.escape(task.customer_unit) if task.customer_unit else "—"

        rows.append(f"""
    <tr>
      <td class="num">{i}</td>
      <td><span class="task-id">{_html.escape(tid)}</span>{quota_mark}</td>
      <td>{_html.escape(task.title)}</td>
      <td>{_category_badge(task.category)}</td>
      <td>{_priority_badge(task.priority)}</td>
      <td class="num"><strong>{task.sp:.1f}</strong></td>
      <td class="num">{rice_cell}</td>
      <td class="num">{task.business_value:.0f}</td>
      <td class="num">{escalation_cell}</td>
      <td>{stage_cell}</td>
      <td>{customer_cell}</td>
    </tr>""")

    util_color = _utilization_color(report.utilization_pct)
    rows_html = "".join(rows)
    return f"""
<h3>Задачи спринта</h3>
<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Task ID</th>
      <th>Название</th>
      <th>Категория</th>
      <th>Приоритет</th>
      <th class="num">SP</th>
      <th class="num">RICE</th>
      <th class="num">Бизнес-ценность</th>
      <th class="num">Эскалации</th>
      <th>Стадия</th>
      <th>Заказчик</th>
    </tr>
  </thead>
  <tbody>{rows_html}
  </tbody>
  <tfoot>
    <tr>
      <td colspan="5">ИТОГО</td>
      <td class="num" style="color:{util_color}">{total_sp:.1f}</td>
      <td colspan="5">из {report.capacity:.1f} SP &nbsp;
        <span style="color:{util_color};font-weight:700">({report.utilization_pct:.1f}%)</span>
      </td>
    </tr>
  </tfoot>
</table>
"""


def _agent_results_section(report: TeamReportData) -> str:
    blocks = []
    for agent_key, label in _AGENT_LABELS.items():
        content = report.agent_results.get(agent_key, "").strip()
        escaped = _html.escape(content) if content else "(нет данных)"
        blocks.append(f"""
<details>
  <summary>{_html.escape(label)}</summary>
  <div class="details-body">
    <pre>{escaped}</pre>
  </div>
</details>""")

    return "<h3>Рекомендации агентов-специалистов</h3>\n" + "\n".join(blocks)


def _iterations_section(report: TeamReportData) -> str:
    if not report.iterations:
        return ""

    items = []
    for rec in report.iterations:
        accepted = rec.validated
        dot_class = "accepted" if accepted else "rejected"
        verdict_class = "ok" if accepted else "fail"
        verdict_text = "ПРИНЯТО" if accepted else "ОТКЛОНЕНО"
        sp_str = f"{rec.total_sp:.1f} SP"

        feedback_block = ""
        if rec.feedback:
            feedback_block = (
                f'<h4>Замечание валидатора</h4>'
                f'<div class="feedback-box">{_html.escape(rec.feedback)}</div>'
            )

        consult_block = ""
        if rec.consultation and rec.consultation.strip() and rec.consultation.strip() != "Нет рекомендаций от агентов.":
            escaped_consult = _html.escape(rec.consultation)
            consult_block = (
                f'<h4>Консультация специалистов (перед следующей итерацией)</h4>'
                f'<div class="consult-box"><pre style="background:transparent;color:var(--text);padding:0">{escaped_consult}</pre></div>'
            )

        plan_escaped = _html.escape(rec.plan.strip()) if rec.plan else "(нет данных)"

        items.append(f"""
<div class="tl-item">
  <div class="tl-dot {dot_class}"></div>
  <details>
    <summary>
      Итерация {rec.iteration}
      <span class="verdict {verdict_class}">{verdict_text}</span>
      &nbsp;<span style="color:var(--muted);font-weight:400;font-size:0.88em">{sp_str}</span>
    </summary>
    <div class="details-body">
      <h4>План критика</h4>
      <pre>{plan_escaped}</pre>
      {feedback_block}
      {consult_block}
    </div>
  </details>
</div>""")

    items_html = "".join(items)
    return f"""
<h3>Ход планирования (итерации критика)</h3>
<details open>
  <summary>Показать все итерации ({len(report.iterations)})</summary>
  <div class="details-body">
    <div class="timeline">
      {items_html}
    </div>
  </div>
</details>
"""


def _team_section(report: TeamReportData) -> str:
    anchor = f'team-{_html.escape(report.team_name)}'
    return f"""
<section id="{anchor}" class="section-anchor">
  <h2>Команда: {_html.escape(report.team_name)}</h2>
  <p class="meta">Session: <code>{_html.escape(report.session_id)}</code></p>
  {_summary_cards(report)}
  {_task_table(report)}
  {_agent_results_section(report)}
  {_iterations_section(report)}
</section>
<hr style="border:none;border-top:1px solid #e2e8f0;margin:40px 0">
"""


def _footer(generated_at: str) -> str:
    return f"""
<footer>
  Sprint Planning Report &middot; {_html.escape(generated_at)} &middot; AI Sprint Planner
</footer>
</body>
</html>
"""


# ── public entry point ─────────────────────────────────────────────────────────

def generate_html_report(team_reports: list[TeamReportData], output_path: str) -> str:
    """Build a self-contained HTML file and write it to output_path. Returns the path."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts = [_header(generated_at)]

    if len(team_reports) > 1:
        parts.append(_toc(team_reports))

    for report in team_reports:
        parts.append(_team_section(report))

    parts.append(_footer(generated_at))

    full = "\n".join(parts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full)

    return output_path
