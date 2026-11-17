from common import (load_context, render_ai_section, render_compare, render_overview,
                    render_overview_charts, render_priority_table)

ctx = load_context(metrics=True, model=True)
render_overview(ctx)
render_overview_charts(ctx)
render_ai_section(ctx)
render_priority_table(ctx)
render_compare(ctx)
