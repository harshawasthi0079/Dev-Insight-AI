from common import load_context, page_header, render_ai_section, render_priority_table

ctx = load_context(model=True)
page_header(ctx, "AI INSIGHTS")
render_ai_section(ctx)
render_priority_table(ctx)
